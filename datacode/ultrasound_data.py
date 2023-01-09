import os, random
from PIL import Image, ImageOps, ImageFilter

from torch import nn, optim
import torch
from torch.utils.data import WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension,  IMG_EXTENSIONS

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

##---------------------- Classes -----------------------------------------------

## Experimental TODO
class MultiImageDataFolders(VisionDataset):
    """ For Loading ImageFolder dataset from Multiple locations
    Modified Class of DatasetFolder
    Args & Attributes: same as ImageFolder of torch.utils
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        print("CHECKER", extensions, is_valid_file)

        super().__init__(root, transform=transform, target_transform=target_transform)

        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.imgs = self.samples

    def find_classes(self, directories: list
            )-> Tuple[List[str], Dict[str, int]]:
        """
        """
        print("#### In Overridden function of Manually Written GetClass ####")

        classes = set()
        for direc in directories: ## Modif
            classes_ = sorted(entry.name for entry in os.scandir(direc) if entry.is_dir())
            if not classes_:
                raise FileNotFoundError(f"Couldn't find any class folder in {direc}.")
            classes.update(classes_)

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    def make_dataset(self, directories: List[str],
                class_to_idx: Optional[Dict[str, int]] = None,
                extensions: Optional[Union[str, Tuple[str, ...]]] = None,
                is_valid_file: Optional[Callable[[str], bool]] = None,
            ) -> List[Tuple[str, int]]:
        """ Generates a list of samples of a form (path_to_sample, class).
        """
        print("#### In Overridden function of Manually Written MakeDataSet ####")

        instances = []
        available_classes = set()

        for directory in directories: ##Modif
            directory = os.path.expanduser(directory)

            if class_to_idx is None:
                _, class_to_idx = self.find_classes(directory)
            elif not class_to_idx:
                raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

            both_none = extensions is None and is_valid_file is None
            both_something = extensions is not None and is_valid_file is not None
            if both_none or both_something:
                raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

            if extensions is not None:

                def is_valid_file(x: str) -> bool:
                    return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

            is_valid_file = cast(Callable[[str], bool], is_valid_file)


            for target_class in sorted(class_to_idx.keys()):
                class_index = class_to_idx[target_class]
                target_dir = os.path.join(directory, target_class)
                if not os.path.isdir(target_dir):
                    continue
                for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        if is_valid_file(path):
                            item = path, class_index
                            instances.append(item)

                            if target_class not in available_classes:
                                available_classes.add(target_class)
        #END loop

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args: index (int): Index
        Returns: tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


## =============================================================================
## Classification


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
    def __init__(self, infer = False):
        self.img_size =  256

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.75, 1.0),
                                interpolation=Image.BICUBIC),
            transforms.RandomAutocontrast(p=0.6),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])

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


def getUSClassifyDataloader(folders, batch_size, workers,
                            balance_class = False,
                            type_=None, # unused
                            ):
    infer_f = False
    shuffle = True
    if type_ in ['valid', 'test', 'infer']:
        infer_f = True
        shuffle = False


    if False and isinstance(folders, list):
        ## TODO:
        is_valid_file = None
        extensions = IMG_EXTENSIONS if is_valid_file is None else None
        dataset = MultiImageDataFolders(folders,
            transform=USClassifyTransform(),
            loader = default_loader,
            extensions = extensions,
            is_valid_file=is_valid_file )
    else:
        dataset = torchvision.datasets.ImageFolder(folders,
                        USClassifyTransform(infer_f))

    # print(dataset.class_to_idx)
    data_info = {"type": type_,
                 "#ClassId": dataset.class_to_idx ,
                 "#DatasetSize": dataset.__len__() }

    sampler = None
    if balance_class:
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