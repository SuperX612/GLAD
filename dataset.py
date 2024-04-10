import os
import os.path
import numpy as np
import PIL.Image as Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from utils import getClassIdx, BatchSampler


class Dataset_from_Image(Dataset):
    def __init__(self, imgs, targets, transform=None):
        self.imgs = imgs  # img paths
        self.targets = targets  # labs is ndarray
        self.transform = transform
        del imgs, targets

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        target = self.targets[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, target


def CelebA_dataset(transform):
    images_all = []
    labels_all = []
    # print("folders:", folders)
    CelebA_path = "../Database/CelebA/Img/img_align_celeba"
    ATTR_DIR = '../Database/CelebA/Anno/identity_CelebA.txt'
    with open(ATTR_DIR, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            filename = info[0]
            filepath_old = os.path.join(CelebA_path, filename)
            images_all.append(filepath_old)
            labels_all.append(int(info[1]) - 1)
    dst = Dataset_from_Image(images_all, np.asarray(labels_all), transform=transform)
    return dst


def lfw_dataset():
    images_all = []
    labels_all = []
    lfw_path = './Database/lfw'
    folders = os.listdir(lfw_path)
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def pubface_dataset():
    images_all = []
    labels_all = []
    pubface_path = './Database/pubface/train'
    folders = os.listdir(pubface_path)
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(pubface_path, fold).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(pubface_path, fold, f))
                labels_all.append(foldidx)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def google_dataset():
    file = open("./Database/google/list_attr.txt", 'r')
    google_path = './Database/google/images'
    folders = os.listdir(google_path)
    images_all = []
    labels_all = []
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(google_path, str(foldidx)).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(google_path, str(foldidx), f))
                labels_all.append(foldidx)
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def ImgNet_dataset():
    images_all = []
    labels_all = []
    ImgNet_path = "./Database/ilsvrc2012/train"
    folders = os.listdir(ImgNet_path)
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(ImgNet_path, fold).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-5:] == '.JPEG':
                images_all.append(os.path.join(ImgNet_path, fold, f))
                labels_all.append(foldidx)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def get_imagenet_exp_dl(batch_size):
    transform = transforms.Compose([
        torchvision.transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()])  #
    train_ds = datasets.ImageFolder('../Database/ilsvrc2012/train', transform=transform)
    val_ds = datasets.ImageFolder("../Database/ilsvrc2012/val", transform=transform)
    train_dl = DataLoader(train_ds, batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True, shuffle=False)
    return train_dl, val_dl


def get_dataloader(datasetname="imagenet", batch_size=8, conflict_num=1, train=False, seed=1234):
    attackeddataset, class_num, channel = getDataset(datasetname, train)
    class_idx = getClassIdx(attackeddataset)
    batchSampler = BatchSampler(class_idx, batch_size, conflict_num, len(attackeddataset), seed=seed, drop_last=True)
    train_loader = torch.utils.data.DataLoader(attackeddataset, batch_sampler=batchSampler)
    return train_loader, class_num


def getDataset(dataname="imagenet", train=True):
    transform = transforms.Compose([
        torchvision.transforms.Resize([224, 224]),
        transforms.ToTensor()])  #
    ''' load data '''
    if dataname == 'imagenet':  # classes:1000, counts:1281167
        if train:
            dst = datasets.ImageFolder('../Database/ilsvrc2012/train', transform=transform)
        else:
            dst = datasets.ImageFolder('../Database/ilsvrc2012/val', transform=transform)
        class_num = 1000
        channel = 3
    elif dataname == "celeba":  # classes:10177, counts:202599
        dst = CelebA_dataset(transform=transform)
        class_num = 10177
        channel = 3
    elif dataname == 'cifar100':  # classes:100, counts:50000
        dst = datasets.CIFAR100('../Database/cifar100', train=train, download=False, transform=transform)
        class_num = 100
        channel = 3
    elif dataname == 'lfw':  # classes:5749, counts:13233
        dst = datasets.LFWPeople('../Database/lfwpeople', download=True, transform=transform)
        class_num = 5749
        channel = 3
    elif dataname == 'cifar10':  # classes:10, counts:50000
        dst = datasets.CIFAR10('../Database/cifar10', download=False, transform=transform)
        class_num = 10
        channel = 3
    elif dataname == "MNIST":  # classes:10, counts:60000
        dst = datasets.MNIST('../Database/MNIST', download=False, transform=transform)
        class_num = 10
        channel = 1
    elif dataname == "FMNIST":  # classes:10, counts:60000
        dst = datasets.MNIST('../Database/Fashion_MNIST', download=False, transform=transform)
        class_num = 10
        channel = 1
    else:
        exit('unknown dataset')
    return dst, class_num, channel

if __name__ == "__main__":
    getDataset("lfw")