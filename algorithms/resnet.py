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

        # Feature Extractor
        self.backbone = self._load_resnet_backbone()
        self.feat_dropout = nn.Dropout(p=self.args.featx_dropout)

        # Classifier
        sizes = [2048] + list(args.classifier)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=self.args.clsfy_dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

        self.classifier = nn.Sequential(*layers)


    def forward(self, x):
        x = self.backbone(x)
        x = self.feat_dropout(x)
        out = self.classifier(x)

        return out


    def _load_resnet_backbone(self):

        torch_pretrain = "DEFAULT" if self.args.featx_pretrain == "DEFAULT" else None


        if self.args.feature_extract == 'resnet18':
            backbone = torchvision.models.resnet50(zero_init_residual=True,
                                 weights=torch_pretrain)
        elif self.args.feature_extract == 'resnet50':
            backbone = torchvision.models.resnet50(zero_init_residual=True,
                                weights=torch_pretrain)

        # Change input Conv
        # conv1_weight = torch.sum(backbone.conv1.weight, axis = 1).unsqueeze(1)
        # backbone.conv1 = nn.Conv2d(1,  64, kernel_size=7, stride=2, padding=3, bias=False)
        # with torch.no_grad():
        #     backbone.conv1.weight.copy_(conv1_weight)


        backbone.fc = nn.Identity() #remove fc of default arch

        # freeze model
        if self.args.featx_freeze:
            print("Freezing Resnet weights ...")
            for param in backbone.parameters():
                param.requires_grad = False

        return backbone