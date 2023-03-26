""" Dataset classes for MBZUAI- BiomedIA Fetal Ultra Sound datasets
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
import torchvision.transforms as torch_transforms
from typing import List, Dict

##---------------------- Generals -----------------------------------------------

def filter_dataframe(self, df, filtering_dict):
    """ Usage:
    {"blacklist":{'class':["4ch"],"machine_type":["Voluson E8","Voluson S10 Expert","V830"]}}
    """

    if "blacklist" in filtering_dict and "whitelist" in filtering_dict:
        raise  Exception("Hey, decide between whitelisting or blacklisting,"+\
                         "Can't do both! remove either one")

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


def get_class_weights(targets, nclasses):
    """
    Sample level weights fro balanced Loss statergy or data sampling
    targets: assumed to be Long ints representing class from dataset
    """

    n_target = len(targets)
    count_per_class = np.zeros(nclasses, dtype=int)
    for c in targets:
        count_per_class[c] += 1
    count_per_class[count_per_class==0] = n_target

    # for passing to Loss funcs
    weight_per_class = np.zeros(nclasses, dtype=float)
    for i in range(nclasses):
        weight_per_class[i] = float(n_target) / float(count_per_class[i])

    # for passing to sampler
    weight_samplewise = np.zeros(n_target, dtype=float)
    for idx, tgt in enumerate(targets):
        weight_samplewise[idx] = weight_per_class[tgt]

    return weight_per_class, weight_samplewise



## =============================================================================
## Classification


class ClassifyDataFromCSV(Dataset):
    def __init__(self, root_folder, csv_path, transform = None,
                        filtering_dict: Dict[str,Dict[str,List]] = None,
                        ):
        """
        """
        self.root_folder = root_folder
        self.df = pd.read_csv(csv_path)

        ## Filter based on some condition in dataframes
        if filtering_dict: self.df = filter_dataframe(self.df, filtering_dict)

        self.class_to_idx ={c:i for i, c in enumerate(sorted(set(
                                    self.df["class"])))}
        self.images_path =  [ os.path.join(root_folder, p)
                             for p in self.df["image_path"] ]
        self.targets =list(map(lambda x: self.class_to_idx[x],
                                    list(self.df["class"]) ))

        if transform: self.transform = transform
        else: self.transform = torch_transforms.ToTensor()


    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        imgpath = self.images_path[index]
        target = self.targets[index]
        image = PIL.Image.open(imgpath).convert("RGB")
        image = self.transform(image)
        return image, target





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
        elif images_folder:  self._imagefolder_handler(images_folder)
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