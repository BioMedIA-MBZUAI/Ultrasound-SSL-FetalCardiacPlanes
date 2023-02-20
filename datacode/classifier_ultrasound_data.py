import os, json
import random
import PIL.Image
import pandas as pd

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


class USClassifyTransform:
    def __init__(self, infer = False, aug_list=[]):
        self.img_size =  256

        trans_ = []
        if "crop" in aug_list: trans_.append( transforms.RandomResizedCrop(
                        self.img_size, scale=(0.75, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC)    )
        if "ctrs" in aug_list: trans_.append(transforms.RandomAutocontrast(p=0.6))
        if "brig" in aug_list: trans_.append(transforms.ColorJitter(brightness=0.5))
        if "affn" in aug_list: trans_.append(transforms.RandomAffine(
                        degrees=(-180, 180), translate=(0.2, 0.2),
                        interpolation=transforms.InterpolationMode.BICUBIC))
        if "pers" in aug_list: trans_.append(transforms.RandomVerticalFlip(p=0.5))

        if "hflp" in aug_list: trans_.append(transforms.RandomHorizontalFlip(p=0.5))
        if "vflp" in aug_list: trans_.append(transforms.RandomVerticalFlip(p=0.5))


        train_transform = transforms.Compose(trans_+[transforms.ToTensor()])

        infer_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        if infer:
            self.transform = infer_transform
        else:
            self.transform = train_transform


    def __call__(self, x):
        y = self.transform(x)
        return y

    def get_composition(self):
        return self.transform



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



## =============================================================================
## Barlow Twin

class USBarlowTwinTransform:
    def __init__(self):
        self.img_size =  256
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.7, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomAffine(degrees=(-180, 180), translate=(0.2, 0.2),
                        interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.5),
            transforms.ToTensor(),

        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.6, 1.0),
                        interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomPerspective(distortion_scale=0.6, p=0.5,
                        interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomAutocontrast(p=0.7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),

        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


def getUSBarlowTwinDataloader(folder, batch_size, workers):

    dataset = torchvision.datasets.ImageFolder(folder, USBarlowTwinTransform()) #train
    data_info = { "#ClassId": dataset.class_to_idx ,
                "#DatasetSize": dataset.__len__() }
    loader = torch.utils.data.DataLoader( dataset, shuffle=True,
                batch_size=batch_size, num_workers=workers,
                drop_last= True, pin_memory=True)

    return loader, data_info



## =============================================================================
## Reconstruction

class USFrameConstruction(Dataset):
    def __init__(self, images_folder, csv_path, transform = None,
                        filtering_dict: Dict[str,Dict[str,List]] = {},
                        ):
        """
        """

        self.images_folder = images_folder
        df = pd.read_csv(csv_path)
        self.df = self._filter_dataframe(df, filtering_dict)

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

