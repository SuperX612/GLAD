import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import nevergrad as ng
from pytorch_pretrained_biggan import (BigGAN, convert_to_images)

class NGReconstructor():
    """
    Reconstruction for BigGAN

    """
    def __init__(self, fl_model, generator, loss_fn, batchsize, num_classes=1000, search_dim=(128,), strategy='CMA', budget=500, use_tanh=True, use_weight=False, defense_setting=None):

        self.generator = generator
        self.budget = budget
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.num_samples = 50
        self.weight = None
        self.defense_setting = defense_setting
        parametrization = ng.p.Array(init=np.random.rand(batchsize, search_dim[0]))
        self.optimizer = ng.optimizers.registry[strategy](parametrization=parametrization, budget=budget)
        self.fl_setting = {'loss_fn':loss_fn, 'fl_model':fl_model, 'num_classes':num_classes}
        if use_weight:
            self.weight = np.ones(62,)
            for i in range(0, 20):
                self.weight[3*i:3*(i+1)] /= 2**i

    def evaluate_loss(self, z, labels, input_gradient):
        return self.ng_loss(z=z, input_gradient=input_gradient, metric='l2',
                        labels=labels, generator=self.generator, weight=self.weight,
                        use_tanh=self.use_tanh, **self.fl_setting, defense_setting=self.defense_setting
                       )

    def reconstruct(self, input_gradient, label, use_pbar=True):
        labels = self.infer_label(input_gradient, num_inputs=len(label))
        # labels = label.detach().clone()
        print('Inferred label: {}'.format(list(labels)))

        if self.defense_setting is not None:
            if 'clipping' in self.defense_setting:
                total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in input_gradient]), 2)
                self.defense_setting['clipping'] = total_norm.item()
                print('Estimated defense parameter: {}'.format(self.defense_setting['clipping']))
            if 'compression' in self.defense_setting:
                n_zero, n_param = 0, 0
                for i in range(len(input_gradient)):
                    n_zero += torch.sum(input_gradient[i]==0)
                    n_param += torch.numel(input_gradient[i])
                self.defense_setting['compression'] = 100 * (n_zero/n_param).item()
                print('Estimated defense parameter: {}'.format(self.defense_setting['compression']))

        c = torch.nn.functional.one_hot(labels, num_classes=self.fl_setting['num_classes']).to(input_gradient[0].device)

        pbar = tqdm(range(self.budget)) if use_pbar else range(self.budget)

        for r in pbar:
            ng_data = [self.optimizer.ask() for _ in range(self.num_samples)]
            loss = [self.evaluate_loss(z=ng_data[i].value, labels=labels, input_gradient=input_gradient) for i in range(self.num_samples)]
            for z, l in zip(ng_data, loss):
                self.optimizer.tell(z, l)

            if use_pbar:
                pbar.set_description("Loss {:.6}".format(np.mean(loss)))
            else:
                print("Round {} - Loss {:.6}".format(r, np.mean(loss)))


        recommendation = self.optimizer.provide_recommendation()
        z_res = torch.from_numpy(recommendation.value).to(input_gradient[0].device)
        if self.use_tanh:
            z_res = z_res.tanh()
        loss_res = self.evaluate_loss(recommendation.value, labels, input_gradient)
        with torch.no_grad():
            x_res = self.generator(z_res.float(), c.float(), 1)
        x_res = nn.functional.interpolate(x_res, size=(224, 224), mode='area')
        x_res = (x_res - torch.min(x_res))/(x_res.max() - torch.min(x_res))
        # x_res[x_res>1] = 1.0
        # x_res[x_res<0] = 0.0
        img_res = convert_to_images(x_res.cpu())

        return z_res, x_res, img_res, loss_res

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):
        last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
        return labels

    @staticmethod
    def ng_loss(z, # latent variable to be optimized
                loss_fn, # loss function for FL model
                input_gradient,
                labels,
                generator,
                fl_model,
                num_classes=1000,
                metric='l2',
                use_tanh=True,
                weight=None, # weight to be applied when calculating the gradient matching loss
                defense_setting=None  # adaptive attack against defense
               ):

        z = torch.Tensor(z).to(input_gradient[0].device)
        if use_tanh:
            z = z.tanh()

        c = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(input_gradient[0].device)

        with torch.no_grad():
            x = generator(z, c.float(), 1)

        x = nn.functional.interpolate(x, size=(224, 224), mode='area')

        # compute the trial gradient
        target_loss = loss_fn(fl_model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters())
        trial_gradient = [grad.detach() for grad in trial_gradient]

        if defense_setting is not None:
            if 'noise' in defense_setting:
                pass
            if 'clipping' in defense_setting:
                trial_gradient = defense.gradient_clipping(trial_gradient, bound=defense_setting['clipping'])
            if 'compression' in defense_setting:
                trial_gradient = defense.gradient_compression(trial_gradient, percentage=defense_setting['compression'])
            if 'representation' in defense_setting: # for ResNet
                mask = input_gradient[-2][0]!=0
                trial_gradient[-2] = trial_gradient[-2] * mask

        if weight is not None:
            assert len(weight) == len(trial_gradient)
        else:
            weight = [1]*len(trial_gradient)

        # calculate l2 norm
        dist = 0
        for i in range(len(trial_gradient)):
            if metric == 'l2':
                dist += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()*weight[i]
            elif metric == 'l1':
                dist += ((trial_gradient[i] - input_gradient[i]).abs()).sum()*weight[i]
        dist /= len(trial_gradient)

        if not use_tanh:
            KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(), axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))
            dist += 0.1*KLD
        return dist.item()

def ggl_algorithm(grad, label, model, device, budget, use_weight, defence_method):
    generator = BigGAN.from_pretrained('biggan-deep-256')
    generator.to(device)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    ng_rec = NGReconstructor(fl_model=model, generator=generator, loss_fn=loss_fn, batchsize=len(label),
                             num_classes=1000, search_dim=(128,), strategy="CMA", budget=budget,
                             use_tanh=True, use_weight=use_weight, defense_setting=defence_method)
    z_res, x_res, img_res, loss_res = ng_rec.reconstruct(grad, label)

    return x_res