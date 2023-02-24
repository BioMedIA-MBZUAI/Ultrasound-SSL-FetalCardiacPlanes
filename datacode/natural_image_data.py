import os, sys, time
import random

import pandas as pd
import PIL

import torch
import torchvision
import torchvision.transforms as torch_transforms
from datacode import augmentations as augs


class Cifar100Dataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, csv_path, transform = None,
                    return_label = False):
        """
        Simple CIFAR Image data loader. image_size:32x32
        return_label: fine_class, coarse_label, none
        """

        self.images_folder = images_folder
        self.df = pd.read_csv(csv_path)

        self._getitem_method = self._get_image_only
        if return_label:
            self._getitem_method = self._get_image_label
            self.label_type = return_label

        if transform: self.transform = transform
        else: self.transform = torch_transforms.ToTensor()

    def _get_image_only(self, index):
        imgpath = os.path.join(self.images_folder, self.df["filename"][index])
        image = PIL.Image.open(imgpath)
        image = self.transform(image)
        return image

    def _get_image_label(self, index):
        imgpath = os.path.join(self.images_folder, self.df["filename"][index])
        image = PIL.Image.open(imgpath)
        image = self.transform(image)
        label = self.df[self.label_type][index]
        return image, label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self._getitem_method(index)

    def get_info(self):
        return {
            "DataSize": self.__len__(),
            "Transforms": str(self.transform),
        }



if __name__ == "__main__":

    traindataset = Cifar100Dataset( images_folder= os.path.join(CFG.datapath,"train_images"),
                                    csv_path= os.path.join(CFG.datapath,"train_list.csv"),
                                    transform = transform_obj)

    validdataset = Cifar100Dataset( images_folder= os.path.join(CFG.datapath,"test_images"),
                                    csv_path= os.path.join(CFG.datapath,"test_list.csv"),
                                    transform = transform_obj)