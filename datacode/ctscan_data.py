import os, random
from PIL import Image, ImageOps, ImageFilter

from torch import nn, optim
import torch
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


class CtClassifyTransform:
    def __init__(self):
        self.img_sz = 32
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        return y1


def getCtClassifyDataloader(folders, batch_size, workers,
                            type_=None, # unused
                            ):
    is_valid_file = None
    extensions = IMG_EXTENSIONS if is_valid_file is None else None
    if isinstance(folders, list):
        ## TODO:
        dataset = MultiImageDataFolders(folders,
            transform=CtClassifyTransform(),
            loader = default_loader,
            extensions = extensions,
            is_valid_file=is_valid_file )
    else:
        dataset = torchvision.datasets.ImageFolder(folders, CtClassifyTransform()) #train

    # print(dataset.class_to_idx)
    data_info = { "#ClassId": dataset.class_to_idx ,
                 "#DatasetSize": dataset.__len__() }

    loader = torch.utils.data.DataLoader( dataset,
                batch_size=batch_size, num_workers=workers,
                drop_last= True, pin_memory=True)

    return loader, data_info