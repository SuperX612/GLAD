import argparse
import copy
from collections import Counter

import torch
from sklearn.metrics import accuracy_score
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader

from dataset import get_dataloader
from main import get_grad_dl
from modellib import chooseAttackedModel, Generator
from utils import BNStatisticsHook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset used to reconstruct", default="cifar10", choices=["imagenet", "celeba", "cifar100"])
    parser.add_argument("--batch_size", help="batch size for training", default=8, type=int)
    parser.add_argument("--conflict_num", default=[1,2,1,1,2,1], help="1~batchsize")
    parser.add_argument("--seed", help="random seeds for experiments", default=5, type=int)
    parser.add_argument("--device", help="which device to use", default="cuda:1")
    parser.add_argument("--defence_method", default=None,
                        choices=['noise', 'clipping', 'compression', 'representation', None])
    return parser.parse_args()

def get_label_stats(gt_label, num_classes):
    LabelCounter = dict(Counter(gt_label.cpu().numpy()))
    labels = list(sorted(LabelCounter.keys()))
    existences = [1 if i in labels else 0 for i in range(num_classes)]
    num_instances = [LabelCounter[i] if i in labels else 0 for i in range(num_classes)]
    num_instances_nonzero = [item[1] for item in sorted(LabelCounter.items(), key=lambda x: x[0])]
    return labels, existences, num_instances, num_instances_nonzero

def sim_iLRG(probs, grad_b, exist_labels, n_images):
    # Solve linear equations to recover labels
    coefs, values = [], []
    # Add the first equation: k1+k2+...+kc=K
    coefs.append([1 for _ in range(len(exist_labels))])
    values.append(n_images)
    # Add the following equations
    for i in exist_labels:
        coef = []
        for j in exist_labels:
            if j != i:
                coef.append(probs[j][i].item())
            else:
                coef.append(probs[j][i].item() - 1)
        coefs.append(coef)
        values.append(n_images * grad_b[i])
    # Convert into numpy ndarray
    coefs = np.array(coefs)
    values = np.array(values)
    # Solve with Moore-Penrose pseudoinverse
    res_float = np.linalg.pinv(coefs).dot(values)
    # Filter negative values
    res = np.where(res_float > 0, res_float, 0)
    # Round values
    res = np.round(res).astype(int)
    res = np.where(res <= n_images, res, 0)
    err = res - res_float
    num_mod = np.sum(res) - n_images
    if num_mod > 0:
        inds = np.argsort(-err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] -= 1
    elif num_mod < 0:
        inds = np.argsort(err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] += 1
    else:
        mod_res = res
    return res, mod_res


def print_util(string, log_file):
    print(string)
    print(string, file=log_file)

def iLRG(probs, grad_b, n_classes, n_images):
    # Solve linear equations to recover labels
    coefs, values = [], []
    # Add the first equation: k1+k2+...+kc=K
    coefs.append([1 for _ in range(n_classes)])
    values.append(n_images)
    # Add the following equations
    for i in range(n_classes):
        coef = []
        for j in range(n_classes):
            if j != i:
                coef.append(probs[j][i].item())
            else:
                coef.append(probs[j][i].item() - 1)
        coefs.append(coef)
        values.append(n_images * grad_b[i].to("cpu"))
    # Convert into numpy ndarray
    coefs = np.array(coefs)
    values = np.array(values)
    # Solve with Moore-Penrose pseudoinverse
    res_float = np.linalg.pinv(coefs).dot(values)
    # Filter negative values
    res = np.where(res_float > 0, res_float, 0)
    # Round values
    res = np.round(res).astype(int)
    res = np.where(res <= n_images, res, 0)
    err = res - res_float
    num_mod = np.sum(res) - n_images
    if num_mod > 0:
        inds = np.argsort(-err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] -= 1
    elif num_mod < 0:
        inds = np.argsort(err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] += 1
    else:
        mod_res = res

    return res, mod_res

def get_irlg_res(cls_rec_probs, b_grad, gt_label, num_classes, num_images, simplified=False):
    labels, existences, num_instances, num_instances_nonzero = get_label_stats(gt_label, num_classes)
    # Recovered Labels
    rec_instances, mod_rec_instances = sim_iLRG(cls_rec_probs, b_grad, labels, num_images) if simplified else iLRG(
        cls_rec_probs,
        b_grad,
        num_classes,
        num_images)
    rec_labels = labels if simplified else list(np.where(rec_instances > 0)[0])
    rec_instances_nonzero = rec_instances if simplified else rec_instances[rec_labels]
    rec_existences = [1 if i in rec_labels else 0 for i in range(num_classes)]
    # Calculate Class-wise Acc, Instance-wise Acc and Recall
    leacc = 1.0 if simplified else accuracy_score(existences, rec_existences)
    lnacc = accuracy_score(num_instances_nonzero if simplified else num_instances, list(rec_instances))
    irec = sum([rec_instances[i] if rec_instances[i] <= num_instances_nonzero[i] else num_instances_nonzero[i] for i in
                range(len(labels))]) / num_images if simplified else sum(
        [rec_instances[i] if rec_instances[i] <= num_instances[i] else num_instances[i] for i in labels]) / num_images
    # Print results
    print('Ground-truth Labels: ', gt_label)
    print('Our Recovered Labels: ', rec_labels)
    prefix = 'Our Recovered Num of Instances by Simplified Method: ' if simplified else 'Our Recovered Num of Instances: '
    print(prefix + ','.join(str(l) for l in list(rec_instances_nonzero)) +
               ' | LnAcc: %.3f | IRec: %.3f' % (
                   lnacc, irec))
    res = [rec_labels, rec_instances_nonzero, rec_instances, existences, mod_rec_instances]
    metrics = [leacc, lnacc, irec]
    return res, metrics

def get_emb(grad_w, grad_b, exp_thre=10):
    # Split scientific count notation
    sc_grad_b = '%e' % grad_b
    sc_grad_w = ['%e' % w for w in grad_w]
    real_b, exp_b = float(sc_grad_b.split('e')[0]), int(sc_grad_b.split('e')[1])
    real_w, exp_w = np.array([float(sc_w.split('e')[0]) for sc_w in sc_grad_w]), \
                    np.array([int(sc_w.split('e')[1]) for sc_w in sc_grad_w])
    # Deal with 0 case
    if real_b == 0.:
        real_b = 1
        exp_b = -64
    # Deal with exponent value
    exp = exp_w - exp_b
    exp = np.where(exp > exp_thre, exp_thre, exp)
    exp = np.where(exp < -1 * exp_thre, -1 * exp_thre, exp)

    def get_exp(x):
        return 10 ** x if x >= 0 else 1. / 10 ** (-x)

    exp = np.array(list(map(get_exp, exp)))
    # Calculate recovered average embeddings for batch_i (samples of class i)
    res = (1. / real_b) * real_w * exp
    res = torch.from_numpy(res).to(torch.float32)
    return res


def post_process_emb(embedding, model, device, alpha=0.01):
    embedding = embedding.to(device)
    # Feed embedding into FC-Layer to get probabilities.
    out = model.fc(embedding) * alpha
    prob = torch.softmax(out, dim=-1)
    return prob


def get_grad_dl(model: nn.Module, dataloader: DataLoader, device):
    model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    hook = BNStatisticsHook(model, train=False)
    # model.eval()
    for x, y in dataloader:
        model.zero_grad()
        hook.clear()
        attacked_x, attack_y = x.to(device), y.to(device)
        attacked_y_pred = model(attacked_x)
        attacked_loss = criterion(attacked_y_pred, attack_y)
        attacked_grad = torch.autograd.grad(attacked_loss, model.parameters())
        grad = [g.detach() for g in attacked_grad]

        pred2 = attacked_y_pred.detach().clone().requires_grad_(True)
        loss2 = criterion(pred2, attack_y)
        loss2.backward()
        dl_dy = pred2.grad

        mean_var_list = hook.mean_var_list
        return attacked_x, attack_y, grad, mean_var_list, attacked_y_pred, dl_dy, attacked_loss


def get_true_label(model, grad, class_num, device, true_label, batch_size):
    cls_rec_probs = []
    w_grad, b_grad = grad[-2], grad[-1]
    for i in range(class_num):
        # Recover class-specific embeddings and probabilities
        cls_rec_emb = get_emb(w_grad[i], b_grad[i])
        # if (not args.silu) and (not args.leaky_relu):
        #     cls_rec_emb = torch.where(cls_rec_emb < 0., torch.full_like(cls_rec_emb, 0), cls_rec_emb)
        # cls_rec_emb = torch.where(w_grad[i] < 0., torch.full_like(w_grad[i], 0), w_grad[i])
        cls_rec_prob = post_process_emb(embedding=cls_rec_emb,
                                        model=model,
                                        device=device,
                                        alpha=1)
        cls_rec_probs.append(cls_rec_prob)
        print(i)
    res, metrics = get_irlg_res(cls_rec_probs=cls_rec_probs,
                                b_grad=b_grad,
                                gt_label=true_label,
                                num_classes=class_num,
                                num_images=batch_size,
                                simplified=False)
    rec_labels, rec_instances_nonzero, rec_instances, existences, mod_rec_instances = res
    recon_label = []
    for i in range(len(rec_labels)):
        for j in range(rec_instances_nonzero[i]):
            recon_label.append(rec_labels[i])
    print(recon_label)
    return torch.tensor(recon_label)


if __name__ == "__main__":
    args = parse_args()
    dataloader, class_num = get_dataloader(datasetname=args.dataset, batch_size=args.batch_size,
                                           conflict_num=args.conflict_num,
                                           train=True, seed=args.seed)
    resnet50 = chooseAttackedModel(modelName="resnet50", pretrained=True, class_num=class_num)
    resnet50 = resnet50.to(args.device)
    x, y, grad, mean_var_list, attacked_y_pred, dl_dy, attacked_loss = get_grad_dl(resnet50, dataloader, args.device)
    get_true_label(resnet50, grad, class_num, args.device, y, args.batch_size)
    # cls_rec_probs = []
    # w_grad, b_grad = grad[-2], grad[-1]
    # for i in range(class_num):
    #     # Recover class-specific embeddings and probabilities
    #     cls_rec_emb = get_emb(w_grad[i], b_grad[i])
    #     # if (not args.silu) and (not args.leaky_relu):
    #     #     cls_rec_emb = torch.where(cls_rec_emb < 0., torch.full_like(cls_rec_emb, 0), cls_rec_emb)
    #     # cls_rec_emb = torch.where(w_grad[i] < 0., torch.full_like(w_grad[i], 0), w_grad[i])
    #     cls_rec_prob = post_process_emb(embedding=cls_rec_emb,
    #                                     model=resnet50,
    #                                     device=args.device,
    #                                     alpha=1)
    #     cls_rec_probs.append(cls_rec_prob)
    #     print(i)
    # res, metrics = get_irlg_res(cls_rec_probs=cls_rec_probs,
    #                             b_grad=b_grad,
    #                             gt_label=y,
    #                             num_classes=class_num,
    #                             num_images=args.batch_size,
    #                             simplified=False)
    # print("finish")