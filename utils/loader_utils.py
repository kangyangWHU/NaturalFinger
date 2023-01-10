__all__=['Cifar10Loader']

import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torchvision import datasets
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional as TF
from PIL import Image
from utils.test_utils import set_seeds

# set_seeds(0)

class DeNormalize(object):
    '''
        denormalize the tensor to [0,1], usage similar to T.Normalize()
    '''
    def __init__(self, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
        self.mean = -np.array(mean)
        self.std = 1/np.array(std)
    def __call__(self, img):
        '''
        :param img: tensor shape [c,h,w]
        :return: tensor with value between [0,1]
        '''
        img = TF.normalize(img, mean=(0,0,0), std=self.std)
        img = TF.normalize(img, mean=self.mean, std=(1,1,1))
        return img


#  differentiable
class ResizePaddingCrop():
    def __init__(self, crop_size, ratio=(0.8, 1.2), value=128):
        self.crop_size = crop_size
        self.ratio = ratio
        self.value = value

    def __call__(self, x):
        assert x.size[-2] == x.size[-1]
        size = x.size[-1]
        min_size, max_size = int(size*self.ratio[0]), int(size*self.ratio[1])
        width = np.random.randint(min_size, max_size)
        height = np.random.randint(min_size, max_size)

        x = TF.resize(x, (height, width),  interpolation=Image.BILINEAR)

        pad_left, pad_right, pad_top, pad_bottom = 0,0,0,0
        if width < self.crop_size:
            w_rem = self.crop_size - width
            pad_left = np.random.randint(0, w_rem)
            pad_right = w_rem - pad_left

        if height < self.crop_size:
            h_rem = self.crop_size - height
            pad_top = np.random.randint(0, h_rem)
            pad_bottom = h_rem - pad_top
        x = TF.pad(x, (pad_left, pad_top, pad_right, pad_bottom), self.value, padding_mode='constant')

        # No need for crop
        if x.size[0] == self.crop_size and x.size[1] == self.crop_size:
            return x
        else:
            i, j, h, w = T.RandomCrop.get_params(x, (self.crop_size, self.crop_size))
            return TF.crop(x, i, j, h, w)

class MyDataSet(Dataset):

    def __init__(self, datas, labels, transform):
        self.transform = transform
        self.datas = datas
        self.labels = labels

    def __getitem__(self, item):

        x = self.datas[item]

        # if x is a image path, load it
        if isinstance(x, tuple):
            x = default_loader(x[0])

        return self.transform(x), self.labels[item]

    def __len__(self):
        return len(self.labels)


class BaseLoader():
    def __init__(self, path, batch_size, train_transforms=None,
                 val_test_transforms=None, num_workers=4, shuffle=True, pin_memory=False, persistent_workers=True, subset="A"):
        self.batch_size = batch_size
        self.path = path
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.train_transforms = train_transforms
        self.val_test_transforms = val_test_transforms
        self.persistent_workers = persistent_workers

        self.subset = subset

        self.train_loaderA = None
        self.train_loaderB = None
        self.finetune_loader = None
        self.test_loader = None

    def set_loader(self, train_data,  train_targets,  test_data, test_targets):
        # trainset.data : numpy format
        dataA, dataB, labelsA, labelsB = train_test_split(train_data,  train_targets,
                                                          train_size=len( train_targets)//2, random_state=0,
                                                          stratify=train_targets)

        val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_targets,
                                                                        train_size=len(test_targets)//2, random_state=0,
                                                                        stratify=test_targets)

        self.train_loaderA = DataLoader(MyDataSet(dataA, labelsA, self.train_transforms),
                                        batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
                                        pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

        self.train_loaderB = DataLoader(MyDataSet(dataB, labelsB, self.train_transforms),
                                        batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
                                        pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

        self.finetune_loader = DataLoader(MyDataSet(val_data, val_labels, self.train_transforms),
                                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle,
                                          pin_memory=False, persistent_workers=self.persistent_workers)

        self.test_loader = DataLoader(MyDataSet(test_data, test_labels, self.val_test_transforms),
                                      batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle,
                                      pin_memory=False, persistent_workers=self.persistent_workers)
    def get_train_loaderA(self):
        return self.train_loaderA
    def get_train_loaderB(self):
        return self.train_loaderB
    def get_finetune_loader(self):
        return self.finetune_loader
    def get_test_loader(self):
        return self.test_loader

# class MNIST_Loader(object):
#
#     def __init__(self, batch_size, train_transforms=None, val_test_transforms=None, shuffle=True):
#         '''
#         :param train_transforms: the transform used on train_mnist set
#         :param val_test_transforms: transforms used on valadition set and test set
#         '''
#
#         self.mnist_path = flags.mnist_path
#         self.batch_size = batch_size
#
#         if train_transforms is None:
#             train_transforms = T.Compose([
#                 T.ToPILImage(),
#                 T.RandomRotation((-15, 15)),
#                 T.ToTensor(),
#                 # T.Normalize(mean=(0.1307,), std=(0.3081,))
#             ])
#
#         if val_test_transforms is None:
#             val_test_transforms = T.Compose([
#                 T.ToTensor()
#             ])
#
#         trainset = datasets.MNIST(root=self.mnist_path, train=True, download=True)
#
#         # split the train_mnist train_data
#         # trainset.data : numpy format
#         train_datas, val_datas, train_labels, val_labels = train_test_split(trainset.data, trainset.targets,
#                                                                             train_size = 50000, random_state=0, stratify=trainset.targets)
#
#         self.train_loader = DataLoader(MyDataSet(train_datas, train_labels, train_transforms),
#                                        batch_size=self.batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
#
#         self.val_loader = DataLoader(MyDataSet(val_datas, val_labels, val_test_transforms),
#                                      batch_size=self.batch_size)
#
#         self.test_loader = DataLoader(datasets.MNIST(root=self.mnist_path, train=False, download=True,
#                                         transform=val_test_transforms), batch_size=self.batch_size, shuffle=shuffle)

class KnockoffCifar100(BaseLoader):

    def __init__(self, path, test_path ,batch_size, img_size=32, train_transforms=None,
                 val_test_transforms=None, num_workers=4, shuffle=True,
                 pin_memory=False, persistent_workers=True, subset="A", **kargs):
        super(KnockoffCifar100, self).__init__(path, batch_size, train_transforms,
                 val_test_transforms, num_workers, shuffle, pin_memory, persistent_workers, subset)
        if self.train_transforms is None:
            self.train_transforms = T.Compose([
                                    # T.ToPILImage(),
                                    # T.Resize(img_size),
                                    T.RandomCrop(img_size, padding=4),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor()
                                ])

        # if test doesn't preprocess, it perform worse
        if self.val_test_transforms is None:
            self.val_test_transforms = T.Compose([
                    T.ToTensor()
                ])

        trainset = datasets.CIFAR100(self.path, train=True, download=False, transform=self.train_transforms)


        if img_size == 64:
            # Testset is Cifar10
            testset = ImageFolder(os.path.join(test_path, "val"), transform=self.val_test_transforms)
        else:
            # Testset is Cifar10
            testset = datasets.CIFAR10(test_path, train=False, download=False, transform=self.val_test_transforms)

        self.train_loaderA = DataLoader(trainset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
                                        pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
        self.train_loaderB = self.train_loaderA
        self.test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
                                        pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

class KnockoffTinyImageNet(BaseLoader):

    def __init__(self, path, test_path, batch_size, img_size=64, train_transforms=None,
                 val_test_transforms=None, num_workers=4, shuffle=True,
                 pin_memory=False, persistent_workers=True, subset="A", **kwargs):
        super(KnockoffTinyImageNet, self).__init__(path, batch_size, train_transforms,
                                                 val_test_transforms, num_workers, shuffle, pin_memory,
                                                 persistent_workers, subset)
        if self.train_transforms is None:
            self.train_transforms = T.Compose([
                # T.Resize(img_size),
                ResizePaddingCrop(crop_size=img_size),
                T.RandomHorizontalFlip(),
                T.RandomChoice([
                    T.ColorJitter(brightness=0.5),
                    T.ColorJitter(contrast=0.5),
                    T.ColorJitter(saturation=0.5),
                    T.ColorJitter(hue=0.5),
                ]),
                T.RandomAffine(degrees=(-15, 15), translate=(0, 0.1)),
                T.ToTensor(),
                T.RandomErasing()
            ])

        # if test doesn't preprocess, it perform worse
        if self.val_test_transforms is None:
            self.val_test_transforms = T.Compose([
                T.ToTensor(),
            ])

        trainset = ImageFolder(os.path.join(self.path, "train"), transform=self.train_transforms)

        if img_size == 64:
            # Testset is Cifar10
            testset = ImageFolder(os.path.join(test_path, "val"), transform=self.val_test_transforms)
        else:
            # Testset is Cifar10
            testset = datasets.CIFAR10(test_path, train=False, download=False, transform=self.val_test_transforms)

        self.train_loaderA = DataLoader(trainset, batch_size=self.batch_size, shuffle=self.shuffle,
                                        num_workers=self.num_workers,
                                        pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
        self.train_loaderB = self.train_loaderA
        self.test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=self.shuffle,
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)


def filter_samples(self, samples, targets, selected_classes, map_list):
        train_samples, train_targets = [], []
        for x, y in zip(samples, targets):
            if y in selected_classes:
                train_samples.append(x)
                train_targets.append(map_list[y])
        return train_samples, train_targets


class Cifar10Loader(BaseLoader):

    def __init__(self, path, batch_size, train_transforms=None,
                 val_test_transforms=None, num_workers=4, shuffle=True,
                 pin_memory=False, persistent_workers=True, subset="A", **kargs):
        super(Cifar10Loader, self).__init__(path, batch_size, train_transforms,
                 val_test_transforms, num_workers, shuffle, pin_memory, persistent_workers, subset)
        if self.train_transforms is None:
            self.train_transforms = T.Compose([
                                    T.ToPILImage(),
                                    T.RandomCrop(32, padding=4),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor()
                                ])

        # if test doesn't preprocess, it perform worse
        if self.val_test_transforms is None:
            self.val_test_transforms = T.Compose([
                    T.ToTensor()
                ])

        trainset = datasets.CIFAR10(self.path, train=True, download=True)
        testset = datasets.CIFAR10(self.path, train=False, download=True)


        self.set_loader(trainset.data,  trainset.targets,  testset.data, testset.targets)



class TinyImagenet50Loader(BaseLoader):

    def __init__(self, path, batch_size, train_transforms=None,
                 val_test_transforms=None, num_workers=4, shuffle=True,
                 pin_memory=False, persistent_workers=True, subset="A", **kwargs):
        super(TinyImagenet50Loader, self).__init__(path, batch_size, train_transforms,
                 val_test_transforms, num_workers, shuffle, pin_memory, persistent_workers, subset)

        if self.train_transforms is None:
            self.train_transforms = T.Compose([
                                    ResizePaddingCrop(crop_size=64),
                                    T.RandomHorizontalFlip(),
                                    T.RandomChoice([
                                        T.ColorJitter(brightness=0.5),
                                        T.ColorJitter(contrast=0.5),
                                        T.ColorJitter(saturation=0.5),
                                        T.ColorJitter(hue=0.5),
                                    ]),
                                    T.RandomAffine(degrees=(-15, 15), translate=(0, 0.1)),
                                    T.ToTensor(),
                                    T.RandomErasing()
            ])

        # if test doesn't preprocess, it perform worse
        if self.val_test_transforms is None:
            self.val_test_transforms = T.Compose([
                    T.ToTensor(),
                ])

        trainset = ImageFolder(os.path.join(self.path, "train"), transform=self.train_transforms)
        testset = ImageFolder(os.path.join(self.path, "val"), transform=self.val_test_transforms)

        self.set_loader(trainset.samples,  trainset.targets,  testset.samples, testset.targets)


class ImagenetLoader(BaseLoader):

    def __init__(self,path, batch_size, num_classes=100, train_transforms=None,
                 val_test_transforms=None, num_workers=4, shuffle=True,
                 pin_memory=False, persistent_workers=True, subset="A", **kwargs):
        super(ImagenetLoader, self).__init__(path, batch_size, train_transforms,
                 val_test_transforms, num_workers, shuffle, pin_memory, persistent_workers, subset)
        self.num_classes = num_classes

        if self.train_transforms is None:
            self.train_transforms = T.Compose([
                T.Resize(224),
                T.RandomCrop(224),
                T.ToTensor()
                ])

        # if test doesn't preprocess, it perform worse
        if self.val_test_transforms is None:
            self.val_test_transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor()
                ])

        trainset = ImageFolder(os.path.join(self.path, "train"), transform=self.train_transforms)
        testset = ImageFolder(os.path.join(self.path, "val"), transform=self.val_test_transforms)

        # set the seeds
        np.random.seed(self.num_classes)
        selected_classes = np.random.choice(range(0, 1000), self.num_classes, replace=False)

        train_samples, train_targets = self.filter_samples(trainset.samples, trainset.targets, selected_classes)
        test_samples, test_targets = self.filter_samples(testset.samples, testset.targets, selected_classes)

        self.set_loader(train_samples, train_targets, test_samples, test_targets)

    def filter_samples(self, samples, targets, selected_classes):
        train_samples, train_targets = [], []
        for x, y in zip(samples, targets):
            if y in selected_classes:
                train_samples.append(x)
                train_targets.append(y)
        return train_samples, train_targets
