import os.path

from matplotlib import pyplot as plt
import numpy as np
import seaborn
import torch
from utils.test_utils import  to_plt_data
from methods.plot_utils import COLORS, set_font_size, MARKER, multi_bars
from methods.metric_utils import read_results

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Times New Roman'

naturalfinger_dir = r"D:\github\FingerprintBench\methods\natural_finger\query_data\cifar10\SAGAN\all\lr=0.5"
metafinger_dir = r"D:\github\FingerprintBench\methods\metafinger\query_data"
# metav_dir = r"D:\github\FingerprintBench\methods\MetaV\query_data"
metav_dir = r"D:\github\FingerprintBench\methods\metav\query_data\random_init\lr=0.001"
ipguard_dir = r"D:\github\FingerprintBench\methods\ipguard\query_data"
test_dir = r"D:\github\FingerprintBench\methods\natural_finger\query_data\cifar10\test"

LEGEND_NAME = ["Ours", "MetaFinger", "MetaV", "IPGuard", "Test"]
TICKS_NAME = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
WN_ticks_name = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]



def show_demo(input_tensor, index, name):
    plt.figure(figsize=(1.5,1.5))
    plt.subplots_adjust(
        top=1.0,
        bottom=0.0,
        left=0.0,
        right=1.0,
        hspace=0.2,
        wspace=0.2
    )
    plt.imshow(to_plt_data(input_tensor))
    plt.axis("off")
    plt.show()
    plt.imsave("plot_figures/cifar10/other_demo/{}_index_{}.pdf".format(name, index), to_plt_data(input_tensor))


def plot_epoch_acc():
    set_font_size(14)
    plt.figure(figsize=(5.5,4))
    plt.subplots_adjust(
        top=0.983,
        bottom=0.141,
        left=0.118,
        right=0.99,
        hspace=0.2,
        wspace=0.2
    )
    plt.grid(ls='--', axis='y')
    acc_list = read_results("results/cifar10/epoch_acc/clean_vgg16_epoch_acc.txt")
    plt.plot(range(len(acc_list)), acc_list, c=COLORS[0], linestyle=None, linewidth=3)

    acc_list = read_results("results/cifar10/epoch_acc/MetaV_vgg16_epoch_acc.txt")
    plt.plot(range(len(acc_list)), acc_list, c=COLORS[1], linestyle=None, linewidth=3)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Test Set", "MetaV Queryset"])
    plt.savefig("plot_figures/epoch_acc_metav.pdf")
    plt.show()



def get_diff(data_dir, pos_slice, neg_slice):
    pos_res = read_results(os.path.join(data_dir, "pos.txt"))
    neg_res = read_results(os.path.join(data_dir, "neg.txt"))
    pos_data = pos_res[pos_slice]
    neg_data = neg_res[neg_slice]
    diff = [pos - neg for pos, neg in zip(pos_data, neg_data)]
    return diff



# adversarial training
def plot_TPR_FPR(data_dir, pos_slice, neg_slice, method_name):

    yticks_name = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    xticks_name = ["0.0","0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    plt.figure(figsize=(5,4))
    set_font_size(16)
    plt.subplots_adjust(
        top=0.97,
        bottom=0.148,
        left=0.139,
        right=0.977,
        hspace=0.2,
        wspace=0.2
    )

    pos_res = read_results(os.path.join(data_dir, "pos.txt"))
    pos_acc = [pos_res[0]] + pos_res[pos_slice]
    neg_res = read_results(os.path.join(data_dir, "neg.txt"))
    neg_acc = [neg_res[10]] + neg_res[neg_slice]

    data_len = len(pos_acc)
    plt.plot(range(data_len), pos_acc, color=COLORS[0],  linewidth=2, marker=MARKER[0])
    plt.plot(range(data_len), neg_acc, color=COLORS[1],linewidth=2, marker=MARKER[1])

    # row
    plt.plot(range(data_len), [min(pos_acc)]*data_len, color="gray", linestyle="--")
    plt.plot(range(data_len), [max(neg_acc)]*data_len, color="gray", linestyle="--")
    plt.xticks(ticks=np.arange(data_len), labels=xticks_name[:data_len])
    plt.yticks(ticks=np.arange(10, 110, 10), labels=yticks_name)

    plt.xlabel("Adversarial Example Ratio", fontsize=16)
    plt.ylabel("Query Set Accuracy")
    plt.legend(["Positive Models", "Negative Models"],
               loc="lower right",
               # bbox_to_anchor=(0.5, 1.15),
               # ncol=2
               )
    plt.savefig("{}/cifar10_{}_AT.pdf".format(data_dir, method_name))
    plt.show()

def plot_AT_ALL():
    pos_slice = slice(35, 45)
    neg_slice = slice(45, 55)
    plot_TPR_FPR(naturalfinger_dir, pos_slice, neg_slice, "naturalfinger")
    plot_TPR_FPR(metafinger_dir, pos_slice, neg_slice, "metafinger")
    # plot_TPR_FPR(metav_dir, pos_slice, neg_slice, "metav")
    # plot_TPR_FPR(ipguard_dir, pos_slice, neg_slice, "ipguard")

plot_AT_ALL()
exit()

# adversarial training
def plot_line(pos_slice, neg_slice, xticks_name, xlabel, name="weight_pruning"):

    yticks_name = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    plt.figure(figsize=(6,5))
    set_font_size(14)
    plt.subplots_adjust(
        top=0.974,
        bottom=0.11,
        left=0.112,
        right=0.985,
        hspace=0.2,
        wspace=0.2
    )
    plt.grid(ls='--', axis='y')
    naturalfinger = get_diff(naturalfinger_dir, pos_slice, neg_slice)
    metafinger = get_diff(metafinger_dir, pos_slice, neg_slice)
    metav = get_diff(metav_dir, pos_slice, neg_slice)
    ipguard = get_diff(ipguard_dir, pos_slice, neg_slice)
    test_data = read_results(os.path.join(test_dir, "pos.txt"))[pos_slice]

    data_len = len(naturalfinger)
    plt.plot(range(data_len), naturalfinger, color=COLORS[0],  linewidth=2, marker=MARKER[0])
    plt.plot(range(data_len), metafinger, color=COLORS[1],linewidth=2, marker=MARKER[1])
    plt.plot(range(data_len), metav, color=COLORS[2],linewidth=2, marker=MARKER[2])
    plt.plot(range(data_len), ipguard, color=COLORS[3], linewidth=2, marker=MARKER[3])
    # plt.plot(range(data_len), test_data, color=COLORS[4], linewidth=2, marker=MARKER[4])

    plt.xticks(ticks=np.arange(data_len), labels=xticks_name[:data_len])
    plt.yticks(ticks=np.arange(10, 110, 10), labels=yticks_name)

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Matching Rate Margin")
    plt.legend(LEGEND_NAME,
               # loc="center",
               # bbox_to_anchor=(0.5, 1.15),
               # ncol=2
               )
    plt.savefig("plot_figures/cifar10/cifar10_{}.pdf".format(name))
    plt.show()

# adversarial training
def plot_bar(pos_slice, neg_slice, xticks_name, xlabel, name="distillation"):
    COLORS = ["#4CD990", "#FF87AD", "#A78CFF", "#55C8F0"]

    # COLORS = ["#55C8F0", "#FF87AD", "#A48CF4", "#96C878", "#FC8D62", "#00B4B4"]
    yticks_name = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

    if name in ["distillation", "knockoff"]:
        plt.figure(figsize=(16,5.5))
        plt.subplots_adjust(
            top=0.985,
            bottom=0.162,
            left=0.044,
            right=0.995,
            hspace=0.2,
            wspace=0.2
        )
    else:
        plt.figure(figsize=(6, 5))
        plt.subplots_adjust(
            top=0.978,
            bottom=0.11,
            left=0.112,
            right=0.985,
            hspace=0.2,
            wspace=0.2
        )

    set_font_size(14)
    plt.grid(ls='--', axis='y')

    naturalfinger = get_diff(naturalfinger_dir, pos_slice, neg_slice)
    metafinger = get_diff(metafinger_dir, pos_slice, neg_slice)
    metav = get_diff(metav_dir, pos_slice, neg_slice)
    ipguard = get_diff(ipguard_dir, pos_slice, neg_slice)
    test_data = read_results(os.path.join(test_dir, "pos.txt"))[pos_slice]
    ipguard = [abs(x) for x in ipguard]
    data = [naturalfinger, metafinger, metav, ipguard]

    color_list = ["white"]*5
    multi_bars(plt, data, xticks_name, colors=COLORS, edgecolors=color_list,  tick_step=1, group_gap=0.3, bar_gap=0)
    plt.yticks(ticks=np.arange(10, 110, 10), labels=yticks_name)
    plt.xticks(ticks=range(len(xticks_name)), labels=xticks_name, fontsize=16)
    # plt.xticks(ticks=range(len(xticks_name)), labels=xticks_name, rotation=15, ha='center')

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Matching Rate Margin")
    plt.legend(LEGEND_NAME,
               # loc="center",
               # bbox_to_anchor=(0.5, 1.15),
               # ncol=2
               )
    plt.savefig("plot_figures/cifar10/cifar10_{}.pdf".format(name))
    plt.show()

def plot_finetuning():

    pos_slice = slice(16, 20)
    neg_slice = slice(26, 30)
    xticks_name = ["FTLL", "FTAL", "RTLL", "RTAL"]

    plot_bar(pos_slice, neg_slice,  xticks_name, xlabel="Fine-Tunging Mode", name="finetuning")

# plot_finetuning()

def plot_distillation():

    pos_slice = slice(1, 16)
    neg_slice = slice(11, 26)
    xticks_name = ["DenseNet121", "EfficientNet", "GoogleNet", "InceptionV3", "MobileNetV2", "ResNet18", "ResNet34", "ResNet50", "SE-ResNet18", "ShuffleNetV2",
               "SqueezeNet", "VGG13_BN", "VGG16_BN", "VGG19_BN", "Xception"]

    plot_bar(pos_slice, neg_slice,  xticks_name, xlabel="Model Architecture", name="distillation")

# plot_distillation()
def plot_model_steal():

    pos_slice = slice(20, 35)
    neg_slice = slice(30, 45)
    xticks_name = ["DenseNet121", "EfficientNet", "GoogleNet", "InceptionV3", "MobileNetV2", "ResNet18", "ResNet34",
                   "ResNet50", "SE-ResNet18", "ShuffleNetV2",
                   "SqueezeNet", "VGG13_BN", "VGG16_BN", "VGG19_BN", "Xception"]

    plot_bar(pos_slice, neg_slice,  xticks_name, xlabel="Model Architecture", name="knockoff")

# plot_model_steal()

def plot_AT():

    pos_slice = slice(35, 45)
    neg_slice = slice(45, 55)

    plot_line(pos_slice, neg_slice, TICKS_NAME, "Adversarial Example Ratio", "AT")

def plot_WN():

    pos_slice = slice(45, 54)
    neg_slice = slice(55, 64)

    plot_line(pos_slice, neg_slice, WN_ticks_name,"Weight Noising Value" ,"WN")

def plot_WP():

    pos_slice = slice(54, 63)
    neg_slice = slice(64, 73)

    plot_line(pos_slice, neg_slice, TICKS_NAME,"Weight Pruning Rate" ,"WP")



def plot_GAN():
    fig = plt.figure(figsize=(5, 4))
    set_font_size(14)

    fig.subplots_adjust(
        top=0.983,
        bottom=0.206,
        left=0.126,
        right=0.973,
        hspace=0.2,
        wspace=0.2
    )
    xticks_name = ["WGAN-GP", "SAGAN", "LOGAN", "ContraGAN", "ICRGAN",  "ReACGAN"]
    # yticks_name = ["0.2", "0.4", "0.6", "0.8", "1.0"]

    data = [0.86,  0.91, 0.88,  0.86, 0.84, 0.8471]
    # multi_bars(plt, data, xticks_name, colors=COLORS, edgecolors=color_list, hatches=None, tick_step=1, group_gap=0.3,
    #            bar_gap=0)
    plt.bar(range(len(data)), data, color=COLORS, width=0.4)
    plt.grid(ls='--', axis='y')
    plt.xticks(ticks=np.arange(len(xticks_name)), labels=xticks_name, rotation=20, ha='center')
    # plt.yticks(ticks=[20, 40, 60, 80, 100], labels=yticks_name)
    plt.xlabel("GAN")
    plt.ylabel("ARUC")

    plt.savefig("plot_figures/cifar10/cifar10_GAN.pdf")
    plt.show()



def plot_detector():
    fig = plt.figure(figsize=(6, 5))
    set_font_size(14)

    # fig.subplots_adjust(
    #     top = 0.971,
    #     bottom = 0.146,
    #     left = 0.148,
    #     right = 0.98,
    #     hspace = 0.2,
    #     wspace = 0.2
    # )
    xticks_name = ["Natural-GP", "SAGAN", "LOGAN", "ContraGAN", "ICRGAN",  "ReACGAN"]
    # yticks_name = ["0.2", "0.4", "0.6", "0.8", "1.0"]

    data = [0.86,  0.91, 0.88,  0.86, 0.84, 0.8471]
    # multi_bars(plt, data, xticks_name, colors=COLORS, edgecolors=color_list, hatches=None, tick_step=1, group_gap=0.3,
    #            bar_gap=0)
    plt.bar(range(len(data)), data, color=COLORS, width=0.5)
    plt.grid(ls='--', axis='y')
    plt.xticks(ticks=np.arange(len(xticks_name)), labels=xticks_name)
    # plt.yticks(ticks=[20, 40, 60, 80, 100], labels=yticks_name)
    plt.xlabel("GAN")
    plt.ylabel("ARUC")

    plt.savefig("plot_figures/cifar10/cifar10_GAN.pdf")
    plt.show()
# plot_GAN()

# plot_WN()
# plot_WP()
#

