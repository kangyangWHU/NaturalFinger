__all__ = [
           "load_model_from_pth", "load_cifar_model", "load_tinyimagenet_model", "load_imagenet_model",
           "set_seeds",'to_plt_data']

import sys
sys.path.append('../')
from torchvision.models import *
from models.cifar10_tinyimagenet import *
import torch
import numpy as np

# fix random seed
def set_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


# turn label to one hot
def to_onehot(label, num_classes):
    batch_size = label.shape[0]
    one_hot = torch.zeros(batch_size, num_classes, device=label.device)
    return one_hot.scatter_(1, label.unsqueeze(dim=1), 1)


def to_infhot(label, num_classes):
    batch_size = label.shape[0]
    one_hot = torch.zeros(batch_size, num_classes, device=label.device)
    # RTLL parameters is value
    return one_hot.scatter_(1, label.unsqueeze(dim=1), float('inf'))


#  minimize loss
class CWLoss(nn.Module):

    def __init__(self, targeted=False, confidence=0, reduction="mean", minimize=True):
        """
        :param targeted:
        :param confidence:
        :param reduction:
        :param minimize: this loss works different with Xent loss in untargeted case
                        if use CWLoss in those attack, set minimize=False
                        cw loss always minimize the loss, xent maximize loss in untargeted case
        """
        super(CWLoss, self).__init__()
        self.targeted = targeted
        self.confidence = confidence
        self.reduction = reduction
        self.minimize = minimize

    def forward(self, logits, label):

        num_classes = logits.shape[1]
        onehot_label = to_onehot(label, num_classes)
        infhot_label = to_infhot(label, num_classes)

        target_logits = logits[onehot_label.bool()]
        other_max_logits = (logits - infhot_label).max(dim=1)[0]

        if self.targeted:
            # this is equal to max(other_max_logits-target_logits, -self.confidence) in math
            # torch.max can't handle this situtaion max( muti-dim-tensor , 0), the clamp works
            loss = torch.clamp(other_max_logits - target_logits, min=-self.confidence)
        else:
            loss = torch.clamp(target_logits - other_max_logits, min=-self.confidence)

            # if used for replace xent loss, you need set
            if not self.minimize:
                loss = -loss

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


def load_model_from_pth(name, path, device, train_flag=False):

    model = eval(name)()
    model.load_state_dict(torch.load(path))
    if train_flag:
        model.train()
    else:
        model.eval()
    model.to(device)
    return model


def load_cifar_model(pth):

    def try_model(model, pth):
        try:
            model.load_state_dict(torch.load(pth))
        except:
            return None
        return model

    if try_model(resnet18_cifar10(), pth):
        return try_model(resnet18_cifar10(), pth)
    elif try_model(resnet34_cifar10(), pth):
        return try_model(resnet34_cifar10(), pth)
    elif try_model(resnet50_cifar10(), pth):
        return try_model(resnet50_cifar10(), pth)
    elif try_model(vgg19_bn_cifar10(), pth):
        return try_model(vgg19_bn_cifar10(), pth)
    elif try_model(vgg9_cifar10(), pth):
        return try_model(vgg9_cifar10(), pth)
    elif try_model(vgg13_bn_cifar10(), pth):
        return try_model(vgg13_bn_cifar10(), pth)
    elif try_model(vgg16_bn_cifar10(), pth):
        return try_model(vgg16_bn_cifar10(), pth)
    elif try_model(googlenet_cifar10(), pth):
        return try_model(googlenet_cifar10(), pth)
    elif try_model(densenet121_cifar10(), pth):
        return try_model(densenet121_cifar10(), pth)
    elif try_model(mobilenetv2_cifar10(), pth):
        return try_model(mobilenetv2_cifar10(), pth)
    elif try_model(squeezenet_cifar10(), pth):
        return try_model(squeezenet_cifar10(), pth)
    elif try_model(shufflenetv2_cifar10(), pth):
        return try_model(shufflenetv2_cifar10(), pth)
    elif try_model(xception_cifar10(), pth):
        return try_model(xception_cifar10(), pth)
    elif try_model(seresnet18_cifar10(), pth):
        return try_model(seresnet18_cifar10(), pth)
    elif try_model(efficientnet_cifar10(), pth):
        return try_model(efficientnet_cifar10(), pth)

    elif try_model(inceptionv3_cifar10(), pth):
        return try_model(inceptionv3_cifar10(), pth)
    else:
        return None

def load_tinyimagenet_model(pth):
    def try_model(model, pth):
        try:
            model.load_state_dict(torch.load(pth))
        except:
            return None
        return model

    if try_model(resnet18_tinyimagenet(), pth):
        return try_model(resnet18_tinyimagenet(), pth)
    elif try_model(resnet34_tinyimagenet(), pth):
        return try_model(resnet34_tinyimagenet(), pth)
    elif try_model(resnet50_tinyimagenet(), pth):
        return try_model(resnet50_tinyimagenet(), pth)
    elif try_model(vgg19_bn_tinyimagenet(), pth):
        return try_model(vgg19_bn_tinyimagenet(), pth)

    elif try_model(vgg13_bn_tinyimagenet(), pth):
        return try_model(vgg13_bn_tinyimagenet(), pth)
    elif try_model(vgg16_bn_tinyimagenet(), pth):
        return try_model(vgg16_bn_tinyimagenet(), pth)
    elif try_model(googlenet_tinyimagenet(), pth):
        return try_model(googlenet_tinyimagenet(), pth)
    elif try_model(densenet121_tinyimagenet(), pth):
        return try_model(densenet121_tinyimagenet(), pth)
    elif try_model(mobilenetv2_tinyimagenet(), pth):
        return try_model(mobilenetv2_tinyimagenet(), pth)

    elif try_model(inceptionv3_tinyimagenet(), pth):
        return try_model(inceptionv3_tinyimagenet(), pth)
    elif try_model(efficientnet_tinyimagenet(), pth):
        return try_model(efficientnet_tinyimagenet(), pth)
    elif try_model(seresnet18_tinyimagenet(), pth):
        return try_model(seresnet18_tinyimagenet(), pth)
    elif try_model(squeezenet_tinyimagenet(), pth):
        return try_model(squeezenet_tinyimagenet(), pth)
    elif try_model(shufflenetv2_tinyimagenet(), pth):
        return try_model(shufflenetv2_tinyimagenet(), pth)
    elif try_model(xception_tinyimagenet(), pth):
        return try_model(xception_tinyimagenet(), pth)
    else:
        return None

def load_imagenet_model(pth):

    def try_model(model, pth):
        try:
            model.load_state_dict(torch.load(pth)["model"])
        except:
            try:
                model.load_state_dict(torch.load(pth))
            except:
                return None
        return model

    if try_model(resnet18(), pth):
        return try_model(resnet18(), pth)
    elif try_model(resnet34(), pth):
        return try_model(resnet34(), pth)
    elif try_model(resnet50(), pth):
        return try_model(resnet50(), pth)
    elif try_model(vgg16_bn(), pth):
        return try_model(vgg16_bn(), pth)
    elif try_model(vgg19_bn(), pth):
        return try_model(vgg19_bn(), pth)
    elif try_model(shufflenet_v2_x1_0(), pth):
        return try_model(shufflenet_v2_x1_0(), pth)
    elif try_model(mobilenet_v2(), pth):
        return try_model(mobilenet_v2(), pth)
    elif try_model(resnext50_32x4d(), pth):
        return try_model(resnext50_32x4d(), pth)
    elif try_model(inception_v3(), pth):
        return try_model(inception_v3(), pth)
    else:
        return None

def to_plt_data(x):
    '''
    convert tensor or numpy (value between [0,1]) to numpy train_data (value between [0,255])
    , which is can be plot, in order to show
    :param x: tensor or numpy
    :type  [c,h,w] without batch  ! important
    :return:
    '''

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    # copy, incase change original train_data
    x = x.copy()
    x *= 255
    x = np.round(x)
    x = x.astype(np.uint8)

    # convert [c, h,w] to [h,w,c]
    if x.shape[0] == 1 or x.shape[0] == 3:
        x = x.transpose(1, 2, 0)

    # convert chanel 1 to 3
    if x.shape[2] == 1:
        x = np.broadcast_to(x, (x.shape[0], x.shape[1], 3))

    return x
