from utils.test_utils import set_seeds, load_cifar_model,load_tinyimagenet_model, to_plt_data, CWLoss
from utils.load_model_utils import load_models, load_models_mp
import os
from utils.defense_utils import *
import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy import stats
from advertorch.attacks import FGSM
from torch import optim
from matplotlib import pyplot as plt

def show_pic(adv):
    n = 0
    for r in range(5):
        for c in range(10):
            plt.subplot(5, 10, n + 1)
            plt.imshow(to_plt_data(adv[n].data))
            n += 1
    plt.show()



class NaturalFinger():
    def __init__(self, dataset, model_root_path, generater_path, discriminator_path, loss_fn, device, num_finger,
                 beta = 0.1, random_samples=False, random_labels=False, input_trans=False):
        self.dataset = dataset
        self.device = device
        self.random_samples=random_samples
        self.random_labels=random_labels
        self.input_trans=input_trans
        self.num_finger = num_finger
        self.beta = beta
        self.generater = self.load_G_or_D(generater_path)
        self.discriminator = self.load_G_or_D(discriminator_path)
        self.loss_fn = loss_fn
        self.setup()

        self.train_neg_models = load_models(os.path.join(model_root_path, "train", "neg"), self.load_model, cuda=True)
        self.train_pos_models = load_models(os.path.join(model_root_path, "train", "pos"), self.load_model, cuda=True)


    def load_G_or_D(self, path):
        generater = torch.load(path)
        generater.eval()
        generater.to(self.device)
        return generater

    def setup(self):
        if self.dataset == "cifar10":
            self.load_model = load_cifar_model
            self.num_classes = 10
        elif self.dataset == "tiny-imagenet":
            self.load_model = load_tinyimagenet_model
            self.num_classes = 50

        if self.input_trans:
            self.trans_list = [
                IdentityMap(),
                HFlip(),
                GaussainNoise(0, 0.02),
                RandTranslate(tx=(0, 5), ty=(0, 5), p=0.5),
                RandShear(shearx=(0, 0.1), sheary=(0, 0.1)),
            ]
        else:
            self.trans_list = [
                IdentityMap(),
            ]

    def cal_loss(self, x, y, model_list):
        correct_list = []
        loss = 0
        for model in model_list:
            logits = model(x)
            loss += self.loss_fn(logits, y)
            correct_list.append((logits.argmax(1) == y).sum().item())

        return loss, correct_list

    def genearte_fake_labels(self, size):
        y_fake = torch.randint(low=0, high=self.num_classes, size=(size,), dtype=torch.long, device=self.device)
        return y_fake
    def generate_fake_images(self, lantents, labels):
        x = self.generater(lantents, labels, eval=True)
        x = (x + 1)/2
        return x

    def get_preds(self, x, model_list, cuda=False):
        preds_list = []
        for model in model_list:

            if cuda:
                model.to(self.device)
                preds = model(x)
                model.cpu()
            else:
                preds = model(x)

            preds_list.append(preds.argmax(1).cpu().numpy())
        preds_list = np.array(preds_list)
        return preds_list

    def epoch_eval(self, x, pos_models, neg_models, y_pos, y_neg, sum_dim=1, cuda=False):

        with torch.no_grad():
            pos_preds = self.get_preds(x, pos_models, cuda)
            neg_preds = self.get_preds(x, neg_models, cuda)

        if isinstance(y_pos, torch.Tensor):
            y_pos = y_pos.cpu().numpy()
        if isinstance(y_neg, torch.Tensor):
            y_neg = y_neg.cpu().numpy()

        pos_acc = (pos_preds == y_pos).sum(sum_dim)
        neg_acc = (neg_preds == y_neg).sum(sum_dim)

        return pos_acc, neg_acc

    def screen_queryset(self, data, pos_dataset, neg_dataset, y_pos, y_neg, TPR=1.0, cuda=False):

        mask = None
        for transform in self.trans_list:
            trans_data = transform(data)
            pos_correct, neg_correct = self.epoch_eval(trans_data, pos_dataset, neg_dataset, y_pos, y_neg, sum_dim=0, cuda=cuda)
            _m = (pos_correct/len(pos_dataset) >= TPR) & (neg_correct/len(neg_dataset) >= TPR)
            if mask is None:
                mask = _m
            else:
                mask &= _m

        # important
        mask &= (y_pos != y_neg).cpu().numpy()

        return data[mask], y_pos[mask], mask

    def select_init_samples(self, z_dim, total_number):
        latents_list,true_labels_list, fake_labels_list = [],[],[]
        batch_size = 200
        total = 0
        while True:
            latents = torch.randn(batch_size, z_dim, dtype=torch.float32, device=self.device)
            fake_labels = self.genearte_fake_labels(batch_size)
            adv = self.generate_fake_images(latents, fake_labels)
            # get labels
            with torch.no_grad():
                pos_preds = self.get_preds(adv, self.train_pos_models)
            true_labels = stats.mode(pos_preds, axis=0)[0][0]
            pos_acc = (pos_preds == true_labels).sum(0)

            # all classifiers predict the same label
            mask = (pos_acc == len(self.train_pos_models))

            if self.random_samples:
                mask[:] = True

            # mask = (pos_acc == (len(self.train_pos_models)+1)//2)
            total += mask.sum()
            latents_list.append(latents[mask])
            true_labels_list.append(torch.tensor(true_labels[mask], dtype=torch.long, device=self.device))
            fake_labels_list.append(fake_labels[mask])

            if total > total_number:
                break
        latents_list = torch.vstack(latents_list)
        true_labels_list = torch.hstack(true_labels_list)
        fake_labels_list = torch.hstack(fake_labels_list)

        return latents_list, true_labels_list, fake_labels_list

    def get_labels(self, adv):
        pos_preds = []
        for model in self.train_pos_models:
            with torch.no_grad():
                logits = model(adv)
            pos_preds.append(logits.argmax(1).data.unsqueeze(0))
        pos_preds = torch.vstack(pos_preds)
        labels = pos_preds.mode(0).values
        return labels

    def get_neg_labels(self, data, labels):
        pos_preds = []
        for model in self.train_pos_models:

            adv = FGSM(model, eps=8/255).perturb(data, labels)
            logits = model(adv)
            pos_preds.append(logits.argmax(1).data.unsqueeze(0))
        pos_preds = torch.vstack(pos_preds)
        labels = pos_preds.mode(0).values
        return labels

    def generate_batch_finger(self, latents, labels, fake_labels, lr, epoches, model_batch=4):


        latents.requires_grad = True
        optimizer = optim.SGD([latents], lr=lr, momentum=0.9, nesterov=True)
        # optimizer = optim.Adam([latents], lr=lr)

        data = self.generate_fake_images(latents, fake_labels)
        show_pic(data)

        # random generate neg labels
        if self.random_labels:
            neg_labels = torch.randint_like(labels, 0, self.num_classes)
        else:
            # generate negative labels with adversarial attacks
            neg_labels = self.get_neg_labels(data, labels)

        for i in range(epoches):

            # requires the same number for negative models and positive models
            pos_index = np.random.permutation(range(len(self.train_pos_models)))
            neg_index = np.random.permutation(range(len(self.train_neg_models)))

            fin_pos_correct_list, fin_neg_correct_list = [], []
            # sample the same number of negative models and positive models
            for start in range(0, len(self.train_pos_models), model_batch):
                _data = self.generate_fake_images(latents, fake_labels)

                # show_pic(_data.data)
                if self.input_trans:
                    idx = np.random.randint(0, len(self.trans_list))
                    data = self.trans_list[idx](_data)
                else:
                    data = _data

                pos_models = [self.train_pos_models[k] for k in pos_index[start:start + model_batch]]
                neg_models = [self.train_neg_models[k] for k in neg_index[start:start + model_batch]]

                optimizer.zero_grad()

                # epoch_acc 25
                pos_loss, pos_correct_list = self.cal_loss(data, labels, pos_models)
                neg_loss, neg_correct_list = self.cal_loss(data, neg_labels, neg_models)

                x = _data*2 - 1
                res_dict = self.discriminator(x,  self.genearte_fake_labels(len(neg_labels)), eval=True)
                d_loss = torch.mean(F.relu(1. + res_dict["adv_output"]))
                loss = pos_loss / len(pos_correct_list) + neg_loss / len(neg_correct_list) + self.beta*d_loss

                loss.backward()
                optimizer.step()

                fin_pos_correct_list.append(pos_correct_list)
                fin_neg_correct_list.append(neg_correct_list)

                # clear cache
                torch.cuda.empty_cache()
            if (i+1) % 10 == 0:
                print("TRAIN, Epoch:{}, Pos acc:{}, Neg acc:{}".format(i, fin_pos_correct_list, fin_neg_correct_list))

        # update the data
        data = self.generate_fake_images(latents, fake_labels)
        query_data, query_labels, mask = self.screen_queryset(data, self.train_pos_models, self.train_neg_models, labels, neg_labels, cuda=False)
        print("Query length:{}".format(len(query_labels)))
        show_pic(data)
        return latents, query_data, query_labels, mask

    def generate_finger(self, lr, batch_size, z_dim, epoches, model_batch=4):

        all_latents, true_labels_list, fake_labels_list = self.select_init_samples(z_dim, self.num_finger*10)

        query_data, query_labels, final_latents_list, original_latents_list = [], [], [],[]
        total = 0
        for start in range(0, len(all_latents), batch_size):
            latents = all_latents[start: min(start+batch_size, len(all_latents))]
            true_labels = true_labels_list[start: min(start+batch_size, len(all_latents))]
            fake_labels = fake_labels_list[start: min(start+batch_size, len(all_latents))]
            final_latents, _data, _labels, mask = self.generate_batch_finger(copy.deepcopy(latents), true_labels, fake_labels, lr, epoches, model_batch)

            total += len(_labels)
            query_data.append(_data.data.cpu())
            query_labels.append(_labels.data.cpu())
            final_latents_list.append(final_latents[mask].data.cpu())
            original_latents_list.append(latents[mask].data.cpu())

            if total >= self.num_finger:
                break

        query_data = torch.vstack(query_data)
        final_latents_list = torch.vstack(final_latents_list)
        original_latents_list = torch.vstack(original_latents_list)
        query_labels = torch.hstack(query_labels)

        torch.save(query_data[:self.num_finger], "query_data/{}-query-data.pt".format(self.dataset))
        torch.save(query_labels[:self.num_finger], "query_data/{}-query-labels.pt".format(self.dataset))

        torch.save(original_latents_list[:self.num_finger], "query_data/{}-original-latents.pt".format(self.dataset))
        torch.save(final_latents_list[:self.num_finger], "query_data/{}-final-latents.pt".format(self.dataset))

if __name__ == '__main__':

    # sample, adv label, input trans, d loss, screen out

    set_seeds(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    batch_size = 80
    z_dim = 128  # ContraGAN 80  WGAN-GP 128  SAGAN 128
    lr = 0.1   # SGD 0.1  Adam
    beta = 0.5
    epoches = 1001
    loss_fn = CWLoss(confidence=50, targeted=True)

    finger = NaturalFinger(dataset="cifar10", model_root_path=r"weights/cifar10",
                           generater_path="GAN_weights/cifar10/SAGAN/CIFAR10-SAGAN-G.pth",
                           discriminator_path="GAN_weights/cifar10/SAGAN/CIFAR10-SAGAN-D.pth",
                           loss_fn=loss_fn, device=device, num_finger=500, beta=beta,
                           random_samples=False, random_labels=False, input_trans=True)

    finger.generate_finger(lr, batch_size, z_dim, epoches, model_batch=4)
