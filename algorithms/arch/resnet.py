import os, sys
import torch
import torchvision
from torch import nn



def loadResnetBackbone(arch, torch_pretrain= None, freeze= False):

    ## pretrain setting
    if torch_pretrain in ["DEFAULT", "IMGNET-1K"]:
        torch_pretrain = "DEFAULT" #"IMAGENET1K_V2"
    elif torch_pretrain in [None, "NONE", "none"]:
        torch_pretrain = None
    else:
        raise ValueError("Unknown pretrain weight type requested ", torch_pretrain )
    print("Torch Pretrain Set to ...", torch_pretrain)

    ## Model loading
    if arch == 'resnet18':
        backbone = torchvision.models.resnet18(zero_init_residual=True,
                                weights=torch_pretrain)
        outfeat_size = 512
    elif arch == 'resnet34':
        backbone = torchvision.models.resnet34(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 512
    elif arch == 'resnet50':
        backbone = torchvision.models.resnet50(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 2048

    elif arch == 'resnet101':
        backbone = torchvision.models.resnet101(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 2048

    elif arch == 'resnet152':
        backbone = torchvision.models.resnet152(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 2048

    else:
        raise ValueError(f"Unknown Model Implementation called in {os.path.basename(__file__)}")
    backbone.fc = nn.Identity() #remove fc of default arch

    # freeze model
    if freeze:
        print("Freezing Resnet weights ...")
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, outfeat_size