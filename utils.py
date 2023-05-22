import os
import copy
import math
import random
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from torchvision import utils as vutils

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compress_one(gradient, compress_rate):
    c = gradient.flatten().abs().sort()[0]
    threshold = c[-int(len(c) * compress_rate)]
    temp = torch.ones(gradient.size())
    temp[gradient.abs() < threshold] = 0
    gradient[gradient.abs() < threshold] = 0
    return gradient, temp


def progress_bar(cur, total):
    total_num = 20
    progress_num = total_num * cur // total
    return progress_num * "#" + (total_num - progress_num) * "=" + f"{100*cur/total:3.1f}%"


def compress_onebyone(gradient, compress_rate):
    gradient = list((_.detach().clone() for _ in gradient))
    mask = []
    for i in range(len(gradient)):
        gradient[i], temp = compress_one(gradient[i], compress_rate)
        mask.append(temp)
    return gradient, mask


def compress_whole(gradients, compress_rate):
    gradients = list((_.detach().cpu().clone() for _ in gradients))
    mask_tuple = []
    c = np.asarray(gradients[0])
    c = abs(c.ravel())
    mask_tuple.append(np.ones(gradients[0].shape))
    for x in gradients[1:]:
        a = np.asarray(x)  # 转化为array
        a = abs(a.ravel())
        c = np.append(c, a)
        mask_tuple.append(np.ones(x.shape))
    sort_c = np.sort(c)
    top = len(sort_c)
    standard = sort_c[int(-top * compress_rate)]
    print('compress shield : ', standard)
    newgra = copy.deepcopy(gradients)
    for i in range(len(newgra)):
        p = np.asarray(newgra[i])
        m = mask_tuple[i]
        m[abs(p) < standard] = 0
        p[abs(p) < standard] = 0
        mask_tuple[i] = torch.tensor(m)
        newgra[i] = torch.tensor(p)
    return newgra, mask_tuple


# def compress_all(gradient, compress_rate):
#     gradient = list((_.detach().clone() for _ in gradient))
#     mask = []
#     if compress_rate < 1.0:
#         c = torch.cat([gradient[i].flatten() for i in range(len(gradient))]).flatten().abs().sort()[0]
#         threshold = c[-int(len(c) * compress_rate)]
#         for i in range(len(gradient)):
#             temp = torch.ones(gradient[i].size())
#             temp[gradient[i].abs() < threshold] = 0
#             mask.append(temp)
#             gradient[i][gradient[i].abs() < threshold] = 0
#     else:
#         for i in range(len(gradient)):
#             temp = torch.ones(gradient[i].size())
#             mask.append(temp)
#     return gradient, mask


def getClassIdx(dataset):
    dic_class = defaultdict(list)
    idx = dataset.targets
    if not isinstance(idx, torch.Tensor):
        idx = torch.Tensor(dataset.targets)
    class_num = int(idx.max() + 1)
    for i in range(class_num):
        dic_class[i] = torch.where(idx == i)[0].tolist()
    return dic_class


class BatchSampler(Sampler):
    # 批次采样
    def __init__(self, class_idx, batch_size, conflict_num, datasetnum, seed=1234, drop_last=True):
        self.class_idx = class_idx
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.datasetnum = datasetnum
        self.conflict_num = conflict_num
        self.seed = seed

    def __iter__(self):
        batch = []
        setup_seed(self.seed)
        for i in range(self.datasetnum // self.batch_size):
            classes = random.sample(range(0, len(self.class_idx)), len(self.conflict_num))
            for j in range(len(self.conflict_num)):
                idx = random.sample(self.class_idx[classes[j]], self.conflict_num[j])
                batch.extend(idx)
            yield batch
            batch = []
        # 如果不需drop最后一组返回最后一组
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.datasetnum // self.batch_size
        else:
            return (self.datasetnum + self.batch_size - 1) // self.batch_size


def show_imgs(imgs, true_imgs, y, save: bool, dir_path, filename, seed, idxx):
    """
    Display of real and generated images, with the option to save as a file or not
    """
    imgs, true_imgs = imgs.detach().cpu(), true_imgs.detach().cpu()
    imgs = imgs.clamp(0, 1)
    imgs = imgs.permute([0, 2, 3, 1])
    true_imgs = true_imgs.clamp(0, 1)
    true_imgs = true_imgs.permute([0, 2, 3, 1])
    img_num = len(true_imgs)
    plt.figure(figsize=[2, img_num])
    plt.suptitle(f"{seed} {idxx}", y=0.99)
    for idx, img in enumerate(true_imgs):
        plt.subplot(img_num, 2, 2 * idx + 1)
        plt.title(str(y[idx].item()), x=0.5, y=1)
        plt.axis("off")
        plt.imshow(img)
    for idx, img in enumerate(imgs):
        plt.subplot(img_num, 2, 2 * idx + 2)
        plt.axis("off")
        plt.imshow(img)
    if save:
        make_dir_if_not_exist(dir_path)
        plt.tight_layout()
        plt.savefig(f"{dir_path}/{filename}.png")
        plt.show()
        plt.close()
    else:
        plt.tight_layout()
        # plt.subplots_adjust(top=0.95)
        plt.show()
    plt.close()


def savecurimgs(x, dummy_x, record_dir, index):
    make_dir_if_not_exist(record_dir + "/" + str(index))
    filedir = record_dir + "/" + str(index) + "/original"
    make_dir_if_not_exist(filedir)
    for i in range(len(x)):
        vutils.save_image(x[i], filedir + "/" + str(i) + ".png")
    filedir = record_dir + "/" + str(index) + "/reconstruct"
    make_dir_if_not_exist(filedir)
    for i in range(len(dummy_x)):
        vutils.save_image(dummy_x[i], filedir + "/" + str(i) + ".png")

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o0777)


def make_reconstructPath(path, filename, make):
    content = set(os.listdir(path))  # 获取目录下列表\
    i = 0
    filenames = filename + '_' + str(i)
    while filenames in content:
        i = i + 1
        filenames = filename + '_' + str(i)
    if make:
        os.makedirs(path + "/" + filenames)
    return path + "/" + filenames


class BNStatisticsHook:
    def __init__(self, model, train=True):
        self.train = train
        self.mean_var_list = []
        self.hook_list = []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = module.register_forward_pre_hook(self.hook_fn)
                self.hook_list.append(hook)

    def hook_fn(self, _, input_data):
        mean = input_data[0].mean(dim=[0, 2, 3])
        var = input_data[0].var(dim=[0, 2, 3])
        if not self.train:
            mean = mean.detach().clone()
            var = var.detach().clone()
        self.mean_var_list.append([mean, var])

    def close(self):
        self.mean_var_list.clear()
        for hook in self.hook_list:
            hook.remove()

    def clear(self):
        self.mean_var_list.clear()


def data_normal(original_img):
    d_min = original_img.min()
    if d_min < 0:
        original_img += torch.abs(d_min)
        d_min = original_img.min()
    d_max = original_img.max()
    dst = d_max - d_min
    norm_data = (original_img - d_min).true_divide(dst)
    return norm_data


if __name__ == "__main__":
    make_reconstructPath("/home/b1107/user/xkl/sameLabelattack/record/reconstruct", "exp")