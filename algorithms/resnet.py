import torch
import torchvision
from torch import nn
import utilities.runUtils as rutl


## Neural Net
class ClassifierNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        rutl.START_SEED()

        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity() #remove fc of default arch

        # Classifier
        sizes = [2048] + list(args.classifier)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        out = self.classifier(x)

        return out