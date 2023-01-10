__all__ = ['efficientnet_tinyimagenet', "efficientnet_cifar10"]

import torch.nn as nn
import torch

def init_linear(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)

class Residual(nn.Module):
    def __init__(self, residual, shortcut=None):
        super().__init__()
        self.shortcut = nn.Identity() if shortcut is None else shortcut
        self.residual = residual
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.shortcut(x) + self.gamma * self.residual(x)

class NormAct(nn.Sequential):
    def __init__(self, channels):
        super().__init__(
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            NormAct(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups),
        )


class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, reduced_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvResidual(nn.Sequential):
    def __init__(self, in_channels, out_channels, expansion, kernel_size=3, stride=1):
        mid_channels = in_channels * expansion
        squeeze_channels = in_channels // 4
        super().__init__(
            ConvBlock(in_channels, mid_channels, 1), # Pointwise
            ConvBlock(mid_channels, mid_channels, kernel_size, stride=stride, groups=mid_channels), # Depthwise
            NormAct(mid_channels),
            SqueezeExciteBlock(mid_channels, squeeze_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1) # Pointwise
        )


class MBConvBlock(Residual):
    def __init__(self, in_channels, out_channels, expansion, kernel_size=3, stride=1):
        residual = MBConvResidual(in_channels, out_channels, expansion, kernel_size, stride)
        shortcut = self.get_shortcut(in_channels, out_channels, stride)
        super().__init__(residual, shortcut)

    def get_shortcut(self, in_channels, out_channels, stride):
        if in_channels != out_channels:
            shortcut = nn.Conv2d(in_channels, out_channels, 1)
            if stride > 1:
                shortcut = nn.Sequential(nn.AvgPool2d(stride), shortcut)
        elif stride > 1:
            shortcut = nn.AvgPool2d(stride)
        else:
            shortcut = nn.Identity()
        return shortcut

class BlockStack(nn.Sequential):
    def __init__(self, num_layers, channel_list, strides, expansion=4, kernel_size=3):
        layers = []
        for num, in_channels, out_channels, stride in zip(num_layers, channel_list, channel_list[1:], strides):
            for _ in range(num):
                layers.append(MBConvBlock(in_channels, out_channels, expansion, kernel_size, stride))
                in_channels = out_channels
                stride = 1
        super().__init__(*layers)

class Head(nn.Sequential):
    def __init__(self, in_channels, classes, mult=4, p_drop=0.):
        mid_channels = in_channels * mult
        super().__init__(
            ConvBlock(in_channels, mid_channels, 1),
            NormAct(mid_channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(mid_channels, classes)
        )

class Stem(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels, stride):
        squeeze_channels = mid_channels // 4
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, 3, stride=stride, padding=1),
            ConvBlock(mid_channels, mid_channels, 3, groups=mid_channels), # Depthwise
            NormAct(mid_channels),
            SqueezeExciteBlock(mid_channels, squeeze_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1) # Pointwise
        )

class EfficientNet(nn.Sequential):
    def __init__(self, classes,  num_layers, channel_list, strides, expansion=4,
                 in_channels=3, head_p_drop=0.):
        super().__init__(
            Stem(in_channels, *channel_list[:2], stride=strides[0]),
            BlockStack(num_layers, channel_list[1:], strides[1:], expansion),
            Head(channel_list[-1], classes, p_drop=head_p_drop)
        )

def efficientnet_tinyimagenet():
    model = EfficientNet(50,
                         num_layers = [4,  4,   4,   4],
                         channel_list = [32, 16, 32, 64, 128, 256],
                         strides = [1,  1,  2,   2,   2],
                         expansion = 4,
                         head_p_drop = 0.3)

    model.apply(init_linear)
    return model

def efficientnet_cifar10():
    model = EfficientNet(10,
                         num_layers = [4,  4,   4,   4],
                         channel_list = [32, 16, 32, 64, 128, 256],
                         strides = [1,  1,  2,   2,   2],
                         expansion = 4,
                         head_p_drop = 0.3)

    model.apply(init_linear)
    return model
