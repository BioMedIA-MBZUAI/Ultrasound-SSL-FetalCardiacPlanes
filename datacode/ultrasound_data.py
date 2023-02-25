""" DataLoader for MBZUAI- BiomedIA Fetal Ultra Sound dataset
"""

import os, sys
import json, glob
import random
import PIL.Image
import h5py
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from typing import List, Dict

##---------------------- Generals -----------------------------------------------

def filter_dataframe(self, df, filtering_dict):

    if "blacklist" in filtering_dict and "whitelist" in filtering_dict:
        raise  Exception("Hey, decide between whitelisting or blacklisting, Can't do both! remove either one")

    if "blacklist" in filtering_dict:
        print("blacklisting...")
        blacklist_dict = filtering_dict["blacklist"]
        new_df = df
        for k in blacklist_dict.keys():
            for val in blacklist_dict[k]:
                new_df = new_df[new_df[k] != val]

    elif "whitelist" in filtering_dict:
        print("whitelisting...")
        whitelist_dict = filtering_dict["whitelist"]
        new_df_list = []
        for k in whitelist_dict.keys():
            for val in whitelist_dict[k]:
                new_df_list.append(df[df[k] == val])
        new_df = pd.concat(new_df_list).drop_duplicates().reset_index(drop=True)

    else:
        print("No filtering of data done, Peace!")
        new_df = df

    return new_df


## =============================================================================
## Classification


class ClassifyDataFromCSV(Dataset):
    def __init__(self, images_folder, csv_path, transform = None,
                        filtering_dict: Dict[str,Dict[str,List]] = {},
                        ):
        """
        """

        self.images_folder = images_folder
        df = pd.read_csv(csv_path)
        self.df = filter_dataframe(df, filtering_dict)

        self.class_to_idx ={c:i for i, c in enumerate(sorted( set(
                                    self.df["class"]  )))}
        self.images_path =  [ os.path.join(images_folder, c, n) for c, n in zip(
                                    self.df["class"], self.df["image_name"]) ]
        self.images_class =list(map(lambda x: self.class_to_idx[x],
                                    list(self.df["class"]) ))

        if transform: self.transform = transform
        else: self.transform = transforms.ToTensor()


    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        imgpath = self.images_path[index]
        label = self.images_class[index]
        image = PIL.Image.open(imgpath)
        image = self.transform(image)
        return image, label



def get_balanced_samples_weight(images, nclasses):
    """ Sample level weights fro balanced sampling statergy
    """
    n_images = len(images)
    count_per_class = [0] * nclasses
    for _, image_class in images:
        count_per_class[image_class] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, (image, image_class) in enumerate(images):
        weights[idx] = weight_per_class[image_class]
    return weights, weight_per_class


def getUSClassifyDataloader(image_folder, csv_file = None,
                            batch_size = 32, workers = 1,
                            filtering_dict = {},
                            balance_class = False,
                            augument_list = [],
                            type_=None, # unused
                            ):
    infer_f = False
    shuffle = True
    if type_ in ['valid', 'test', 'infer']:
        infer_f = True
        shuffle = False

    transforms = USClassifyTransform(infer_f, augument_list)

    if csv_file:
        dataset = ClassifyDataFromCSV( image_folder, csv_file,
                                        transforms, filtering_dict )
    else:
        dataset = torchvision.datasets.ImageFolder(image_folder, transforms)
        if filtering_dict: raise Exception("BlackListing and Filtering data is not implemented for ImageFolder structure based Loading")

    # print(dataset.class_to_idx)
    data_info = {"type": type_,
                 "#ClassId": dataset.class_to_idx ,
                 "#DatasetSize": dataset.__len__(),
                 "#Transforms": str(transforms.get_composition()),
                 "#Filtering": filtering_dict,
                 }

    sampler = None
    if balance_class: #TODO: Fix for Custom Dataloader
        samples_per_epoch = int(1.5 * len(dataset.imgs))
        s_weight, freq = get_balanced_samples_weight(dataset.imgs, len(dataset.classes))
        sampler = WeightedRandomSampler(s_weight, samples_per_epoch)
        data_info["WeightedSampler"] = freq
        shuffle = False

    loader = torch.utils.data.DataLoader( dataset, shuffle = shuffle,
                batch_size=batch_size, num_workers=workers, sampler = sampler,
                drop_last= True, pin_memory=True)

    return loader, data_info


##================ US Video Frames Loader ======================================


class FetalUSFramesDataset(torch.utils.data.Dataset):
    """ Treats Video frames as Independant images for trainng purposes
    """
    def __init__(self, images_folder=None, hdf5_file=None,
                        transform = None,
                        load2ram = False, frame_skip=None):
        """
        """
        self.load2ram = load2ram
        self.frame_skip = frame_skip
        #tobedefined
        self.image_paths= []
        self.image_frames= []
        self.get_image_func = None
        ##-----

        if transform: self.transform = transform
        else: self.transform = torch_transforms.ToTensor()

        if hdf5_file:       self._hdf5file_handler(hdf5_file)
        elif image_folder:  self._imagefolder_handler(images_folder)
        else: raise Exception("No Data info to load")


    # for image folder handling
    def _imagefolder_handler(self, images_folder):
        def __get_image_lazy(index):
            return PIL.Image.open(self.image_paths[index]).convert("RGB")
        def __get_image_eager(index):
            return self.image_frames[index]

        self.image_paths = sorted(glob.glob(images_folder+"/**/*.png"))

        self.get_image_func = __get_image_lazy
        if self.load2ram:
            self.image_frames = [ __get_image_lazy(i)
                                    for i in range(len(self.image_paths))]
            self.get_image_func = __get_image_eager

        print("Frame Skip is not implemented")

    # for hdf5 file handling
    def _hdf5file_handler(self, hdf5_file):
        def __get_image_lazy(index):
            k, i = self.image_paths[index]
            arr = self.hdfobj[k][i]
            return PIL.Image.fromarray(arr).convert("RGB")

        def __get_image_eager(index):
            return self.image_frames[index]

        self.hdfobj = h5py.File(hdf5_file,'r')
        for k in self.hdfobj.keys():
            for i in range(self.hdfobj[k].shape[0]):
                if i % self.frame_skip: continue
                self.image_paths.append([k, i])

        self.get_image_func = __get_image_lazy
        if self.load2ram:
            self.image_frames = [ __get_image_lazy(i)
                                    for i in range(len(self.image_paths))]
            self.get_image_func = __get_image_eager



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.get_image_func(index)
        image = self.transform(image)
        return image

    def get_info(self):
        print(self.get_image_func)
        return {
            "DataSize": self.__len__(),
            "Transforms": str(self.transform),
        }