import os
from PIL import Image
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms


## GradCam and other Attention
## https://github.com/jacobgil/pytorch-grad-cam
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCamVisualizer:

    def __init__(self, model, visfolder, transform = None,
                    grad_cam = {"layers": None, "class_target": None},
                    device = "cpu"):
        self.device = device
        if not os.path.exists(visfolder): os.makedirs(visfolder)
        self.visfolder = visfolder

        if transform: self.transforms = transforms
        else: transform =self.transforms = transforms.Compose([transforms.ToTensor()])

        self.cam_list = []
        for lyr in grad_cam["layers"]:
            self.cam_list.append(
                GradCAM(model=model, target_layers=grad_cam["layers"],
                        use_cuda=device)   )

        cls_tgt = grad_cam["class_targets"]
        self.targets = [ClassifierOutputTarget(cls_tgt) ] if cls_tgt else None


    def grad_cam_logger(self, image_path, ):


        img, tnsr = self._image_loader()
        out_list = []
        for cam in self.cam_list:
            grayscale_cam = cam(input_tensor=tnsr, targets=self.targets)
            grayscale_cam = grayscale_cam[0, :]
            imposed_img = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            out_list.append()


    def _image_loader(self, image_name):
        """load image, returns cuda tensor"""

        image = Image.open(image_name)
        tensor = self.transforms(image).float()
        tensor = Variable(image, requires_grad=True)
        tensor = tensor.unsqueeze(0)  #for batch Size
        return image, tensor