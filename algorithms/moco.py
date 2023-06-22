import torch
import copy
from torch import nn, optim
from typing import List, Optional, Tuple, Union

from algorithms.arch.resnet import loadResnetBackbone
import utilities.runUtils as rutl

def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)

def deactivate_requires_grad(params):
    """Deactivates the requires_grad flag for all parameters.

    """
    for param in params:
        param.requires_grad = False

##==================== Model ===============================================

class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.
    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).
    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """

    def __init__(
        self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super(ProjectionHead, self).__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.
        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        return self.layers(x)

class MoCoProjectionHead(ProjectionHead):
    """Projection head used for MoCo.

    "(...) we replace the fc head in MoCo with a 2-layer MLP head (hidden layer
    2048-d, with ReLU)" [0]

    [0]: MoCo, 2020, https://arxiv.org/abs/1911.05722

    """

    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 128):
        super(MoCoProjectionHead, self).__init__([
            (input_dim, hidden_dim, None, nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])

class MoCo(nn.Module):
    def __init__(self, featx_arch, pretrained=None, backbone=None):
        super().__init__()

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone, outfeatx_size = loadResnetBackbone(arch=featx_arch,
                                    torch_pretrain=pretrained)
        self.projection_head = MoCoProjectionHead(outfeatx_size, 2048, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum.parameters())
        deactivate_requires_grad(self.projection_head_momentum.parameters())

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key