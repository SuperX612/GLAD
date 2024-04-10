import copy
import csv
import os
import random
import time
from collections import defaultdict

import numpy as np
import piqa
import torch
import argparse

import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Sampler, Dataset
from torchvision import transforms, datasets, models

from RGLA_algorithm import gen_attack_
from defense import gradient_clipping, defense
from gen_attack import gen_attack_algorithm
from modellib import chooseAttackedModel, Generator, Generator336

from optim_attack import ig_algorithm, stg_algorithm
from optim_attack.dlg import dlg_algorithm
from optim_attack.ggl_mulbatch import ggl_algorithm
from utils import setup_seed, show_imgs, BNStatisticsHook, make_reconstructPath, savecurimgs
from dataset import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights", help="weights path for generator", default="./savedModel/RGLA_generator_336.pth") # ./record/train_generator/exp_0/weights_6.pth, ./savedModel/gen_weights.pth
    parser.add_argument("--fgla_modelpath", help="", default="./savedModel/gen_weights.pth")
    parser.add_argument("--reconstruct_num", help="number of reconstructed batches", default=20, type=int)
    parser.add_argument("--algorithm", default="RGLA", choices=["dlg", "ig", "ggl", "fgla", "RGLA", "stg"])
    parser.add_argument("--dataset", help="dataset used to reconstruct", default="imagenet", choices=["imagenet", "celeba", "cifar100"])
    parser.add_argument("--max_iteration", help="iteration to reconstruct", default=20000, type=int)
    parser.add_argument("--reconstructPath", help="experiment name used to create folder", default="./record/reconstruct")
    parser.add_argument("--batch_size", help="batch size for training", default=8, type=int)
    parser.add_argument("--device", help="which device to use", default="cuda:1")
    parser.add_argument("--seed", help="random seeds for experiments", default=77, type=int)
    parser.add_argument("--conflict_num", default=[], help="1~batchsize")
    parser.add_argument("--save_rec", default=False)
    parser.add_argument("--Iteration", default=20001)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--trueloss", default=False, help="")
    parser.add_argument("--trueyhat", default=False, help="")
    # defence methods
    parser.add_argument("--defence_method", default=None, choices=['noise', 'clipping', 'compression', 'representation', None])
    parser.add_argument("--d_param", default=0)
    # for ggl
    parser.add_argument("--budget", default=500)
    parser.add_argument('--use_weight', action='store_true')
    return parser.parse_args()

def save_history(index, pred_loss, psnr, ssim, lpips, time, dir_path):
    with open(f"{dir_path}/history.csv", "a") as f:
        f.write(f"{index}, {pred_loss}, {psnr}, {ssim}, {lpips}, {time}\n")

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
            # classes.sort()
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

def get_imagenet_exp_dl(batch_size):
    transform = transforms.Compose([
        torchvision.transforms.Resize([336, 336]),
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

def getDataset(dataname="imagenet", train=True):
    transform = transforms.Compose([
        torchvision.transforms.Resize([336, 336]),
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

def get_grad_dl(model: nn.Module, dataloader: DataLoader, device, defence_method, d_param):
    model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    hook = BNStatisticsHook(model, train=False)
    decoder = Generator()
    decoder.load_state_dict(torch.load(args.model_weights, map_location=args.device))
    # model.eval()
    for x, y in dataloader:
        model.zero_grad()
        hook.clear()
        attacked_x, attack_y = x.to(device), y.to(device)
        attacked_y_pred = model(attacked_x)
        attacked_loss = criterion(attacked_y_pred, attack_y)
        attacked_grad = torch.autograd.grad(attacked_loss, model.parameters())
        grad = [g.detach() for g in attacked_grad]
        grad = defense(defence_method, grad, model, x, y, d_param)

        # generator = Generator().to(device)
        # weight_inv = torch.pinverse(model.fc.weight.data.transpose(0, -1))
        # fcin = torch.mm(attacked_y_pred - model.fc.bias.data, weight_inv)  # .transpose(0, -1)
        # # fcin = (fcin - fcin.min())/(fcin.max() - fcin.min())
        # reimgs = generat_img(generator, fcin, device)
        # show_imgs(reimgs, x, y, save=args.save_rec, dir_path=f"{record_dir}/all/", filename=f"{index}", seed=args.seed,
        #           idxx=index)

        pred2 = attacked_y_pred.detach().clone().requires_grad_(True)
        loss2 = criterion(pred2, attack_y)
        loss2.backward()
        dl_dy = pred2.grad

        # g = grad[-2]
        # w = model.fc.weight.data.transpose(0, -1)
        # ydldy1 = torch.mm(dl_dy.transpose(0, -1), pred2 - model.fc.bias.data)
        # ydldy2 = torch.mm(g, w)
        # dl_dy_inv = torch.pinverse(dl_dy)
        # model.fc = nn.Sequential()
        # fcin1 = model(attacked_x)
        # # tt = torch.mm(dl_dy.transpose(0, -1), fcin)
        # decoder = Generator()
        # decoder.load_state_dict(torch.load(args.model_weights))
        # fcin2 = torch.mm(g.transpose(0, -1), dl_dy_inv).transpose(0, -1)
        # decoder = decoder.to(device)
        # reimgs1 = decoder(fcin1)
        # reimgs2 = decoder(fcin2)
        # show_imgs(reimgs1.detach().cpu(), reimgs2.detach().cpu(), y, save=args.save_rec, dir_path=f"{record_dir}/images/", filename=f"{index}")

        mean_var_list = hook.mean_var_list
        yield attacked_x, attack_y, grad, mean_var_list, attacked_y_pred, dl_dy, attacked_loss

def Resnet18(pretrained=False, class_num=1000):
    model = models.resnet18(pretrained=pretrained)
    model.avgpool = nn.Sequential()
    model.fc = nn.Linear(512 * 7 * 7, class_num)
    return model


def Resnet50(pretrained=False, class_num=1000):
    model = models.resnet50(pretrained=pretrained)
    model.avgpool = nn.Sequential()
    model.fc = nn.Linear(247808, class_num)
    # model.load_state_dict(torch.load("./record/train_model/resnet50_cifar100_1.0.pth"))
    return model


def vgg16(pretrained=False, class_num=1000):
    model = models.vgg16(pretrained=pretrained)
    model.fc = nn.Linear(2048 * 7 * 7, class_num)
    return model


def chooseAttackedModel(modelName="resnet18", pretrained=False, class_num=1000):
    setup_seed(1234)
    if modelName=="resnet18":
        return Resnet18(pretrained=pretrained, class_num=class_num)
    elif modelName=="resnet50":
        return Resnet50(pretrained=pretrained, class_num=class_num)
    elif modelName=="vgg16":
        return vgg16(pretrained=pretrained, class_num=class_num)
    else:
        exit("wrong attacked model")


if __name__ == "__main__":
    args = parse_args()
    record_dir = make_reconstructPath(args.reconstructPath,  str(args.algorithm) + "_" + str(args.dataset) + "_batchsize" + str(args.batch_size) + "_defense" + str(args.defence_method) + "_dpara" + str(args.d_param) + "_seed" + str(args.seed) + "_exp_", make=args.save_rec)
    if args.dataset not in ["imagenet", "celeba", "cifar100", "cifar10"]:
        raise Exception(f"can not find dataset {args.dataset}!")
    if sum(args.conflict_num) > args.batch_size:
        raise Exception("can not conflict so much")
    if torch.tensor(args.conflict_num).sum() > args.batch_size:
        raise  Exception("conflict num is larger than the batchsize")
    while torch.tensor(args.conflict_num).sum() < args.batch_size:
        args.conflict_num.append(1)
    dataloader, class_num = get_dataloader(datasetname=args.dataset, batch_size=args.batch_size, conflict_num=args.conflict_num,
                                train=True, seed=args.seed)
    resnet50 = chooseAttackedModel(modelName="resnet50", pretrained=True, class_num=class_num)
    grad_dl = get_grad_dl(resnet50, dataloader, args.device, args.defence_method, args.d_param)
    index = 0
    psnr_lossfn = piqa.PSNR().to(args.device)
    ssim_lossfn = piqa.SSIM().to(args.device)
    lpips_lossfn = piqa.LPIPS().to(args.device)
    defense_setting = dict()
    defense_setting[args.defence_method] = args.d_param
    for x, y, grad, mean_var_list, attacked_y_pred, dl_dy, attacked_loss in grad_dl:
        index += 1
        start=time.time()
        if args.algorithm == "dlg":
            dummy_x = dlg_algorithm(grad, y, resnet50, (args.batch_size, 3, 336, 336), args.max_iteration, args.device, class_num)
        elif args.algorithm == "ig":
            dummy_x = ig_algorithm(grad, x, y, resnet50, (args.batch_size, 3, 336, 336), args.max_iteration,
                                   args.device, record_dir)
        elif args.algorithm == "ggl":
            dummy_x = ggl_algorithm(grad, y, resnet50, args.device, args.budget, args.use_weight, defense_setting)
        elif args.algorithm == "fgla":
            decoder = Generator()
            decoder.load_state_dict(torch.load(args.fgla_modelpath))
            dummy_x = gen_attack_algorithm(grad, y, decoder, True, args.device)
        elif args.algorithm == "stg":
            dummy_x = stg_algorithm(grad, y, mean_var_list, resnet50, (args.batch_size, 3, 336, 336),
                                    args.max_iteration,
                                    args.device)
        else:
            decoder = Generator336()
            decoder.load_state_dict(torch.load(args.model_weights, map_location=args.device))
            dummy_x, predloss = gen_attack_(grad, y, args.batch_size, resnet50, decoder, args.device, class_num, args.lr, args.Iteration, attacked_y_pred, dl_dy, attacked_loss, args.trueloss, args.trueyhat, args.defence_method, args.d_param)
        psnr = psnr_lossfn(dummy_x, x)
        ssim = ssim_lossfn(dummy_x, x)
        lpips = lpips_lossfn(dummy_x, x)
        if args.save_rec:
            if args.algorithm == "RGLA":
                save_history(index, predloss, psnr, ssim, lpips, time.time() - start, f"{record_dir}")
            else:
                save_history(index, 0, psnr, ssim, lpips, time.time() - start, f"{record_dir}")
            torch.save(dummy_x, f"{record_dir}/dummy_x_{index}.pth")
            savecurimgs(x, dummy_x, record_dir, index)
        print(f"[idx:{index}] ", "psnr:{:.5f}, ssim:{:.5f}, lpips:{:.5f}, time:{}".format(psnr, ssim, lpips, time.time() - start))
        show_imgs(dummy_x, x, y, save=args.save_rec, dir_path=f"{record_dir}/all/", filename=f"{index}", seed=args.seed, idxx=index)
        if index >= args.reconstruct_num:
            break
    print("\nexp finish!")