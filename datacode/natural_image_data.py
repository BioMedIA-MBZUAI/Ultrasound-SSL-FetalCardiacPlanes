import random
from PIL import Image, ImageOps, ImageFilter

from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms

def getCifar100Dataloader(folder, batch_size, workers, type_ = 'train'):

    if type_ == 'train': trainset = True
    else: trainset = False
    dataset = torchvision.datasets.CIFAR100(folder, train=trainset, download=True,
                transform= transforms.ToTensor(), )
    cls_idx = dataset.class_to_idx
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=workers,
        pin_memory=True)

    return loader, cls_idx




