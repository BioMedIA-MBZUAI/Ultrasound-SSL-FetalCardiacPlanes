import os, sys
import math
import torch
from torch import nn, optim
from torch.nn import functional as torch_F

sys.path.append(os.getcwd())
from algorithms.arch.resnet import loadResnetBackbone

## Codes from VIC-Reg official implementation with distributed training blocks removed
##==================== Model ===============================================

class VICReg(nn.Module):
    def __init__(self, featx_arch, projector_sizes,
                    batch_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0,
                    featx_pretrain=None,
                 ):
        super().__init__()

        self.sim_coeff    = sim_coeff
        self.std_coeff    = std_coeff
        self.cov_coeff    = cov_coeff
        self.batch_size   = batch_size

        self.num_features = projector_sizes[-1]
        self.backbone, out_featx_size = loadResnetBackbone(
                            arch=featx_arch,torch_pretrain=featx_pretrain)
        self.projector = self.load_ProjectorNet(out_featx_size, projector_sizes)


    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = torch_F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(torch_F.relu(1 - std_x)) / 2 + \
                    torch.mean(torch_F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + self.off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

    def load_ProjectorNet(self, outfeatx_size, projector_sizes):
        # backbone_out_shape + projector_dims
        sizes = [outfeatx_size] + list(projector_sizes)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        projector = nn.Sequential(*layers)
        return projector

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



##==================== OPTIMISER ===============================================

class LARS(optim.Optimizer):
    def __init__( self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
            weight_decay_filter=True,  lars_adaptation_filter=True,
                ):
        defaults = dict( lr=lr, weight_decay=weight_decay,
            momentum=momentum, eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        ## BT uses seperate params handling of weights and biases here
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    ## BT does notn bother base LR
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    ## Handles weights and Biases seperately
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr




##==================== DEBUG ===============================================

if __name__ == "__main__":

    from torchinfo import summary

    model = VICReg( featx_arch='efficientnet_b0',
                    projector_sizes=[8192,8192,8192],
                    batch_size = 4,
                    featx_pretrain=None)
    summary(model, [(16, 3, 200, 200), (16, 3, 200, 200)])
    # print(model)