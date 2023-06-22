import random
import numpy as np
from PIL import ImageOps, ImageFilter, Image
import torchvision.transforms as torch_transforms
from torchvision.transforms import InterpolationMode

# import phasepack

##========================Natural Images========================================

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

###-----------------------------------------------------------------------------

## For SimCLR
class SimCLRTransform:
    """Transforms for SimCLR during training step of the pre-training stage.
    Transform::
        RandomResizedCrop(size=self.image_size)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        RandomApply([GaussianBlur(kernel_size=int(0.1 * self.image_size))], p=0.5)
        transforms.ToTensor()

    Example::

        transform = SimCLRTrainDataTransform(image_size=32)
        x = sample()
        (xi, xj, xk) = transform(x) # xk is only for the online evaluator if used
    """

    def __init__(
        self, image_size: int = 256, jitter_strength: float = 1.0, normalize=None
    ) -> None:



        self.jitter_strength = jitter_strength
        self.image_size = image_size
        self.normalize = normalize

        kernel_size = int(0.1 * self.image_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.data_transform = torch_transforms.Compose([
            torch_transforms.RandomResizedCrop(size=self.image_size),
            torch_transforms.RandomHorizontalFlip(p=0.5),
            torch_transforms.RandomApply([
                    torch_transforms.ColorJitter(
                            0.8 * self.jitter_strength,
                            0.8 * self.jitter_strength,
                            0.8 * self.jitter_strength,
                            0.2 * self.jitter_strength)], p=0.8),
            torch_transforms.RandomGrayscale(p=0.2),
            torch_transforms.RandomApply(
                    [torch_transforms.GaussianBlur(kernel_size=kernel_size)],
                    p=0.5),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, sample):
        transform = self.data_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj

    def get_composition(self):
        return self.data_transform



## For BalowTwins and VICRegularization
class BarlowTwinsTransformOrig:
    def __init__(self, image_size = 256):
        self.transform = torch_transforms.Compose([
            torch_transforms.RandomResizedCrop(image_size,
                                    interpolation=InterpolationMode.BICUBIC),
            torch_transforms.RandomHorizontalFlip(p=0.5),
            torch_transforms.RandomApply(
                [torch_transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
                p=0.8
            ),
            torch_transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = torch_transforms.Compose([
            torch_transforms.RandomResizedCrop(image_size,
                                    interpolation=InterpolationMode.BICUBIC),
            torch_transforms.RandomHorizontalFlip(p=0.5),
            torch_transforms.RandomApply(
                [torch_transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
                p=0.8
            ),
            torch_transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

    def get_composition(self):
        return (self.transform, self.transform_prime)


##======================== UltraSound Images ===================================


class ClassifierTransform:
    def __init__(self, mode = "train", image_size = 256):

        data_mean = [0.485, 0.456, 0.406]
        data_std  = [0.229, 0.224, 0.225]

        train_transform = torch_transforms.Compose([
            torch_transforms.Resize(image_size,
                                    interpolation=InterpolationMode.BICUBIC),
            torch_transforms.RandAugment(num_ops=5, magnitude=5),
            torch_transforms.RandomHorizontalFlip(p=0.5),
            torch_transforms.RandomVerticalFlip(p=0.5),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=data_mean, std=data_std)
        ])
        infer_transform = torch_transforms.Compose([
            torch_transforms.Resize(image_size,
                        interpolation=InterpolationMode.BICUBIC),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=data_mean, std=data_std)
        ])

        if   mode == "train": self.transform = train_transform
        elif mode == "infer": self.transform = infer_transform
        else : raise ValueError("Unknown Mode set only `train` or `infer` allowed")


    def __call__(self, x):
        y = self.transform(x)
        return y

    def get_composition(self):
        return self.transform

##------------------------------------------------------------------------------


# def phasecongruence(image):
#     image = np.asarray(image)
#     [M, ori, ft, T] = phasepack.phasecongmono(image,
#                             nscale=5, minWaveLength=5)
#     out = ((M - M.min())/(M.max() - M.min()+1) *255.0).astype(np.uint8)
#     out = Image.fromarray(out).convert("RGB")
#     return out


class CustomInfoMaxTransform:
    def __init__(self, image_size = 256):

        self.transform = torch_transforms.Compose([
            torch_transforms.RandomResizedCrop(256,
                                    interpolation=InterpolationMode.BICUBIC),
            torch_transforms.RandomHorizontalFlip(p=0.5),
            # torch_transforms.Lambda(phasecongruence),
            torch_transforms.RandAugment(num_ops=5, magnitude=5),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = torch_transforms.Compose([
            torch_transforms.RandomResizedCrop(image_size,
                                    interpolation=InterpolationMode.BICUBIC),
            torch_transforms.RandomHorizontalFlip(p=0.5),
            torch_transforms.RandomApply(
                [torch_transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
                p=0.8
            ),
            torch_transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

    def get_composition(self):
        return (self.transform, self.transform_prime)


##------------------------------------------------------------------------------



class AEncStandardTransform:
    def __init__(self, image_size = 256):

        data_mean = [0.485, 0.456, 0.406]
        data_std  = [0.229, 0.224, 0.225]

        train_transform = torch_transforms.Compose([
            torch_transforms.Resize(image_size,
                                    interpolation=InterpolationMode.BICUBIC),
            torch_transforms.RandAugment(num_ops=3, magnitude=3),
            torch_transforms.RandomHorizontalFlip(p=0.5),
            torch_transforms.RandomVerticalFlip(p=0.5),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=data_mean, std=data_std)
        ])

        self.transform = train_transform


    def __call__(self, x):
        y = self.transform(x)
        return y, y

    def get_composition(self):
        return self.transform



class AEncInpaintTransform:
    def __init__(self, image_size = 256):

        data_mean = [0.485, 0.456, 0.406]
        data_std  = [0.229, 0.224, 0.225]

        self.transform = torch_transforms.Compose([
            torch_transforms.Resize(image_size,
                                    interpolation=InterpolationMode.BICUBIC),
            torch_transforms.RandAugment(num_ops=1, magnitude=1),
            torch_transforms.RandomHorizontalFlip(p=0.5),
            torch_transforms.RandomVerticalFlip(p=0.5),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=data_mean, std=data_std)
        ])

        self.erase = torch_transforms.RandomErasing(p=1, value="random")


    def __call__(self, x):
        y2 = self.transform(x)
        y1 = self.erase(y2)
        return y1, y2

    def get_composition(self):
        return (self.transform, self.erase)