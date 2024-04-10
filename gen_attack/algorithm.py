import torch


def gen_attack_algorithm(grad, y, generator, offset: bool, device):
    conv_out = flg(grad, y, offset)
    conv_out, decoder = conv_out.to(device), generator.to(device)
    imgs = generator(conv_out)
    return imgs


def flg(grad, y, offset: bool):
    bz = len(y)
    g_w = grad[-2]
    g_b = grad[-1]
    # g_w = g_w + torch.normal(mean=0.,std=0.3,size=g_w.size()).to(g_w.device)
    # g_b = g_b + torch.normal(mean=0., std=0.3, size=g_b.size()).to(g_w.device)
    if offset:
        offset_w = torch.stack([g for idx, g in enumerate(g_w) if idx not in y], dim=0).mean(dim=0) * (bz - 1) / bz
        offset_b = torch.stack([g for idx, g in enumerate(g_b) if idx not in y], dim=0).mean() * (bz - 1) / bz
        conv_out = (g_w[y] - offset_w) / (g_b[y] - offset_b).unsqueeze(1)
    else:
        conv_out = g_w[y] / g_b[y].unsqueeze(1)
    conv_out[torch.isnan(conv_out)] = 0.
    conv_out[torch.isinf(conv_out)] = 0.
    return conv_out
