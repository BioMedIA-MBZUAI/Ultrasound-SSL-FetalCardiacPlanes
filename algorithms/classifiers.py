import os, sys
import torch
import torchvision
from torch import nn

sys.path.append(os.getcwd())
from algorithms.arch.resnet import loadResnetBackbone
import utilities.runUtils as rutl


##================= CLassifier Wrapper =========================================

class ClassifierNet(nn.Module):
    def __init__(self, arch, fc_layer_sizes=[512,1000],
                    feature_dropout=0, classifier_dropout=0,
                    torch_pretrain=None):
        super().__init__()
        rutl.START_SEED(7)

        self.fc_layer_sizes = fc_layer_sizes

        # Feature Extractor
        self.backbone,self.feat_outsize = loadResnetBackbone(arch=arch,
                                            torch_pretrain=torch_pretrain )
        self.fx_dropoutlayer = nn.Dropout(p=feature_dropout)

        # Classifier
        sizes = [self.feat_outsize] + list(self.fc_layer_sizes)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=classifier_dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

        self.classifier = nn.Sequential(*layers)


    def forward(self, x):
        x = self.backbone(x)
        x = self.fx_dropoutlayer(x)
        out = self.classifier(x)

        return out



if __name__ == "__main__":

    from torchinfo import summary

    model = ClassifierNet(arch='efficientnet_b0', fc_layer_sizes=[64,8],
                    feature_dropout=0, classifier_dropout=0,
                    torch_pretrain=None)
    summary(model, (1, 3, 200, 200))
    print(model)