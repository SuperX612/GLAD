import copy
import csv
import os
import time
import piqa
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader

from RGLA_algorithm import gen_attack_
from defense import gradient_clipping, defense
from gen_attack import gen_attack_algorithm
from modellib import chooseAttackedModel, Generator
from optim_attack import ig_algorithm, stg_algorithm, idlg_algorithm
from optim_attack.dlg import dlg_algorithm
from optim_attack.ggl_mulbatch import ggl_algorithm
from utils import setup_seed, show_imgs, BNStatisticsHook, make_reconstructPath, savecurimgs
from dataset import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights", help="weights path for generator", default="./savedModel/RGLA_generator_224.pth") # ./record/train_generator/exp_0/weights_6.pth, ./savedModel/gen_weights.pth
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

        pred2 = attacked_y_pred.detach().clone().requires_grad_(True)
        loss2 = criterion(pred2, attack_y)
        loss2.backward()
        dl_dy = pred2.grad

        mean_var_list = hook.mean_var_list
        yield attacked_x, attack_y, grad, mean_var_list, attacked_y_pred, dl_dy, attacked_loss


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
            dummy_x = idlg_algorithm(grad, y, resnet50, (args.batch_size, 3, 224, 224), args.max_iteration, args.device)
        elif args.algorithm == "ig":
            dummy_x = ig_algorithm(grad, x, y, resnet50, (args.batch_size, 3, 224, 224), args.max_iteration,
                                   args.device, record_dir)
        elif args.algorithm == "ggl":
            dummy_x = ggl_algorithm(grad, y, resnet50, args.device, args.budget, args.use_weight, defense_setting)
        elif args.algorithm == "fgla":
            decoder = Generator()
            decoder.load_state_dict(torch.load(args.fgla_modelpath))
            dummy_x = gen_attack_algorithm(grad, y, decoder, True, args.device)
        elif args.algorithm == "stg":
            dummy_x = stg_algorithm(grad, y, mean_var_list, resnet50, (args.batch_size, 3, 224, 224),
                                    args.max_iteration,
                                    args.device)
        else:
            decoder = Generator()
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