import datetime
import os
import torch.distributed as dist
import piqa
import pyiqa
import torch
import argparse

import torchvision
from torch import nn
from torchvision import datasets, transforms

from dataset import get_imagenet_exp_dl
from modellib import chooseAttackedModel, Generator
from utils import setup_seed, make_dir_if_not_exist, show_imgs


def train_generator(result_dir, batch_size, epochs, generator, origin_model, device):
    origin_model.fc = nn.Sequential()  # remove linear layer
    origin_model = origin_model.to(device)
    generator = generator.to(device)
    criterion = nn.MSELoss()
    psnr_lossfn = piqa.PSNR().to(device)
    ssim_lossfn = piqa.SSIM().to(device)
    lpips_lossfn = piqa.LPIPS().to(device)
    train_dl, val_dl = get_imagenet_exp_dl(batch_size)
    optim = torch.optim.Adam(generator.parameters(), lr=0.00000001)
    # ExpLR = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)
    for epoch in range(0, epochs):
        start = datetime.datetime.now()
        generator.train()
        train_loss = 0
        for idx, (img, y) in enumerate(train_dl):
            img = img.to(device)
            optim.zero_grad()
            pre_img = generator(origin_model(img))
            # loss = criterion(pre_img, img)
            psnr = psnr_lossfn(pre_img, img)
            ssim = ssim_lossfn(pre_img, img)
            lpips = lpips_lossfn(pre_img, img)
            loss = - psnr + 10 * ( 1 - ssim) + 10 * lpips
            loss.backward()
            optim.step()
            if idx % 10 == 0:
                show_training_state(idx, len(train_dl), loss.item(), psnr.item(), ssim.item(), lpips.item(), True, datetime.datetime.now() - start)
            if idx % 1000 == 0:
                save_weights(epoch, generator, result_dir=result_dir)
                show_imgs(pre_img.detach().cpu(), img.detach().cpu(), y, save=False, dir_path=f"{result_dir}/images/", filename=f"{epoch}_train", seed=args.seed)
                # ExpLR.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_dl)
        pre_img, img = pre_img.detach().cpu(), img.detach().cpu()
        show_imgs(pre_img, img, y, save=False, dir_path=f"{result_dir}/images/", filename=f"{epoch}_train", seed=args.seed)

        generator.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, (img, y) in enumerate(val_dl):
                img = img.to(device)
                pre_img = generator(origin_model(img))
                # loss = criterion(pre_img, img)
                psnr = psnr_lossfn(pre_img, img)
                ssim = ssim_lossfn(pre_img, img)
                lpips = lpips_lossfn(pre_img, img)
                loss = - psnr + 10 * (1 - ssim) + 10 * lpips
                if idx % 10 == 0:
                    show_training_state(idx, len(val_dl), loss.item(), psnr.item(), ssim.item(), lpips.item(), False, datetime.datetime.now() - start)
                val_loss += loss.item()
        val_loss = val_loss / len(val_dl)
        pre_img, img = pre_img.detach().cpu(), img.detach().cpu()
        show_imgs(pre_img, img, y, save=False, dir_path=f"{result_dir}/images/", filename=f"{epoch}_val", seed=args.seed)
        show_final_state(epoch, train_loss, val_loss, datetime.datetime.now() - start)
        save_history(epoch, train_loss, val_loss, psnr.item(), ssim.item(), lpips.item(), result_dir=result_dir)
        save_weights(epoch, generator, result_dir=result_dir)


def save_history(epoch, train_loss, val_loss, psnr, ssim, lpips, result_dir):
    """
    Save the losses from the training process to a csv file
    """
    make_dir_if_not_exist(result_dir)
    with open(f"{result_dir}/history.csv", "a") as f:
        f.write(f"{epoch}, {train_loss}, {val_loss}, {psnr}, {ssim}, {lpips}\n")


def show_training_state(idx, length, loss, psnr, ssim, lpips, train: bool, time):
    print("\r" + " " * 50, end="")
    status = "validating"
    if train:
        status = "training"
    print("\r{} [step:{}/{}] loss:{:.3f}, psnr:{:.3f}, ssim{:.3f}, lpips{:.3f}, {}".format(status, idx, length, loss, psnr, ssim, lpips, time), end="")


def show_final_state(epoch, train_loss, val_loss, time):
    print("\r" + " " * 50, end="")
    print(f"\repoch:{epoch} train_loss:{train_loss} val_loss:{val_loss} {time}", end="\n")


def save_weights(epoch, model, result_dir):
    make_dir_if_not_exist(result_dir)
    torch.save(model.state_dict(), f"{result_dir}/weights_{epoch}.pth")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", help="experiment name used to create folder", default="exp_1")
    parser.add_argument("--batch_size", help="batch size for training", default=32, type=int)
    parser.add_argument("--epochs", help="epochs for training", default=3, type=int)
    parser.add_argument("--device", help="which device to use", default="cuda:1")
    parser.add_argument("--seed", help="random seeds for experiments", default=1234, type=int)
    parser.add_argument("--model_weights", help="weights path for generator", default="./savedModel/FGLA_generator.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_dir = f"./record/train_generator/{args.exp_name}/"
    make_dir_if_not_exist(record_dir)
    setup_seed(args.seed)
    resnet50 = chooseAttackedModel(modelName="resnet50", pretrained=True, class_num=1000)
    generator = Generator()
    generator.load_state_dict(torch.load("./savedModel/FGLA_generator.pth"))

    train_generator(record_dir,
                    args.batch_size,
                    args.epochs,
                    generator,
                    resnet50,
                    args.device)

    print(f"finish training, see result in {record_dir}")
