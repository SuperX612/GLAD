import torch
from torch import nn

from defense import defense
from gen_attack import gen_attack_algorithm


def cal_label_dict(label):
    label_dict = {}
    for i in label:
        label_dict[i.item()] = (label==i).sum().item()
    return label_dict

def pred_loss(grad, label, batchsize, defence_method):
    bz = len(label)
    g_b = grad[-1]
    # offset_b = torch.stack([g for idx, g in enumerate(g_b) if idx not in label], dim=0).mean() * (bz - 1) / bz
    gb = g_b[label] # - offset_b
    # noise_norm = torch.tensor(0.01)
    # gb = gb + noise_norm
    pred_loss = torch.tensor(0.0)
    label_dict = cal_label_dict(label)
    count = 0
    for i in range(batchsize):
        # print(torch.log(gb[i] + (label_dict[label[i].item()] / batchsize)))
        tt = gb[i] + (label_dict[label[i].item()] / batchsize)
        if tt.item() <= 0:
            pass
            # pred_loss = pred_loss - torch.log(gb.mean() + torch.tensor(0.01).to(gb.device))
        else:
            count += 1
            pred_loss = pred_loss - torch.log(tt)
    return pred_loss / count

def get_w_lambda(ydldy_target_size, label, device):
    w_lambda = torch.full(ydldy_target_size, 1.0)
    label_dict = cal_label_dict(label)
    for i in label_dict:
        w_lambda[i] = 1.0  # 0.1
    w_lambda = w_lambda.to(device)
    return w_lambda

def get_pred(grad, label, model, batchsize, class_num, device, lr, Iteration, attacked_loss, trueloss, defence_method, generator):
    model.to(device)
    # init_x = gen_attack_algorithm(grad, label, generator, True, device)
    # init_output = model(init_x)
    pred_modelPred = torch.randn((batchsize, class_num)).to(device).requires_grad_(True) # torch.randn((batchsize, class_num)).to(device).requires_grad_(True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([pred_modelPred], lr=lr)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    ydldy_target = torch.mm(grad[-2], model.fc.weight.data.transpose(0, -1))
    if trueloss:
        pred_modelloss = attacked_loss
    else:
        pred_modelloss = pred_loss(grad, label, batchsize, defence_method)
    pred_modelloss = pred_modelloss.detach().clone()
    # fu_offset = fuoffset(label, batchsize)

    # w_lambda = torch.full(ydldy_target.size(),1.0)
    # w_lambda[label] = 1.0 # 0.1
    # w_lambda = w_lambda.to(device)
    w_lambda = get_w_lambda(ydldy_target.size(), label, device).detach().clone()

    for iter in range(Iteration):
        optimizer.zero_grad()
        predloss = criterion(pred_modelPred, label)
        predloss.backward(retain_graph=True)
        dldy = pred_modelPred.grad
        pred_dldy = dldy.detach().clone()
        ydldy = torch.mm(dldy.transpose(0, -1), pred_modelPred - model.fc.bias.data)
        # ydldy = defense("clipping", [ydldy], model, 0, label, 8)[0]
        w_loss = torch.mul(w_lambda, ydldy - ydldy_target).pow(2).sum()
        b_loss = (torch.sum(dldy, 0) - grad[-1]).pow(2).sum()
        loss_loss = (predloss - pred_modelloss).pow(2).sum()
        # w_loss = torch.abs(torch.mul(w_lambda, ydldy - ydldy_target)).sum()
        # b_loss = torch.abs(torch.sum(dldy, 0) - grad[-1]).sum()
        # loss_loss = torch.abs(predloss - pred_modelloss).sum()
        # loss_loss = ((torch.abs(8.0 - predloss) + torch.abs(predloss - 6.0)) - torch.tensor(2.0)).pow(2)
        loss = 10000 * w_loss + b_loss + loss_loss
        loss.backward()
        optimizer.step()
        if iter % 1000 == 0:
            ExpLR.step()
            print("{}/{} loss: {:.5f}, w_loss: {}, b_loss: {}, loss_loss: {}".format(iter, Iteration, loss.item(), w_loss.item(), b_loss.item(), loss_loss.item())) # b_loss: {b_loss.item()},
    # pred_dldy = torch.mm(ydldy_target, torch.pinverse(pred_modelPred - model.fc.bias.data)).transpose(0, -1)
    return pred_modelPred, pred_dldy, pred_modelPred

def adjustdldy(dummy_dl_dy, pred_modelPred, attacked_y_pred):
    order_idx = []
    idx_set = set(range(len(dummy_dl_dy)))
    for i in range(len(attacked_y_pred)):
        most_match_idx = list(idx_set)[0]
        min_distance = (pred_modelPred[most_match_idx] - attacked_y_pred[i]).pow(2).sum()
        for j in idx_set:
            cur_distance = (pred_modelPred[j] - attacked_y_pred[i]).pow(2).sum()
            if cur_distance < min_distance:
                most_match_idx = j
                min_distance = cur_distance
        idx_set.remove(most_match_idx)
        order_idx.append(most_match_idx)
    return dummy_dl_dy[torch.tensor(order_idx)], pred_modelPred[torch.tensor(order_idx)]


def generat_img(generator, fcin, device):
    if device == "cpu" or device == "cuda:0":
        device = "cuda:1"
    generator, fcin = generator.to(device), fcin.to(device)
    reimgs = []
    if len(fcin) > 10:
        for i in range(len(fcin) // 10):
            slice = generator(fcin[i * 10: i * 10 + 10])
            reimgs.append(slice.detach().cpu())
        slice = generator(fcin[i*10 + 10:])
    else:
        slice = generator(fcin)
    reimgs.append(slice.detach().cpu())
    return torch.cat(reimgs)

def gen_attack_(grad, label, batchsize, model, generator: nn.Module, device, class_num, lr, Iteration, attacked_y_pred, dl_dy, attacked_loss, trueloss: bool, trueoutput: bool, defence_method, d_param):
    model.to(device)
    if not trueoutput:
        pred_modelPred, dldy, pred_modelPred = get_pred(grad, label, model, batchsize, class_num, device, lr, Iteration, attacked_loss, trueloss, defence_method, generator)
        dldy, pred_modelPred = adjustdldy(dldy, pred_modelPred, attacked_y_pred)
        predloss = (attacked_y_pred - pred_modelPred).pow(2).sum() / len(label)

        print(f"pred loss: {predloss}")
        # for t in range(len(label)):
        #     fig, ax = plt.subplots()  # 创建图实例
        #     ax_x = np.linspace(0, class_num -1, class_num)  # 创建x的取值范围
        #     ax.plot(ax_x, np.array(pred_modelPred[t].detach().cpu()), label='dummy')
        #     ax.plot(ax_x, np.array(attacked_y_pred[t].detach().cpu()), label='true')
        #     ax.legend()  # 自动检测要在图例中显示的元素，并且显示
        #     plt.show()

        print(f"dl_dy loss: {torch.abs(dldy - dl_dy).sum()}")
    else:
        dldy = dl_dy
        predloss = 0
    dl_dy_inv = torch.pinverse(dldy)
    if defence_method == "clipping":
        total_norm = torch.tensor(100.5) # torch.norm(torch.stack([torch.norm(g, 2.0).to(device) for g in grad]), 2.0)
        clip_coef = float(d_param) / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        dw = grad[-2].div(clip_coef_clamped).transpose(0,-1)
        fcin = torch.mm(dw, dl_dy_inv).transpose(0, -1)
        # fcin = torch.mm(grad[-2].transpose(0, -1), dl_dy_inv).transpose(0, -1)
    else:
        fcin = torch.mm(grad[-2].transpose(0, -1), dl_dy_inv).transpose(0, -1)
    # weight_inv = torch.pinverse(model.fc.weight.data.transpose(0, -1))
    # fcin = torch.mm(attacked_y_pred - model.fc.bias.data, weight_inv) # .transpose(0, -1)
    reimgs = generat_img(generator, fcin, device)
    return reimgs.to(device), predloss
    # single_label_imgs = single_label_imgs_find(grad, x, label, model, generator, offset, device)
    # show_imgs(single_label_imgs.detach().cpu(), x.detach().cpu(), label, save=args.save_rec, dir_path=f"{record_dir}/images/", filename=f"{index}")
    # conflict_label_imgs = conflict_label_imgs_find(grad, label, model, generator, offset, device, class_num, single_label_imgs, batchsize)
    # return torch.cat((single_label_imgs, conflict_label_imgs), 0)