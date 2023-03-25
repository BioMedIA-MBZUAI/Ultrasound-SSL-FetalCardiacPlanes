import random
from PIL import ImageOps, ImageFilter
import torchvision.transforms as torch_transforms
from torchvision.transforms import InterpolationMode

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


## For balowTwins and VICRegularization
class BarlowTwinsTransformOrig:
    def __init__(self, image_size = 224):
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
            torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
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
            torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

    def get_composition(self):
        return (self.transform, self.transform_prime)


##======================== UltraSound Images ===================================
