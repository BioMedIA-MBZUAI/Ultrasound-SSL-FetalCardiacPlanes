import os, sys
import torch
import torchvision
from torch import nn

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl


## Neural Net
class ClassifierNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        rutl.START_SEED()

        self.args = args

        # Feature Extractor
        self.backbone, self.feat_outsize = self._load_resnet_backbone()
        self.feat_dropout = nn.Dropout(p=self.args.featx_dropout)

        # Classifier
        sizes = [self.feat_outsize] + list(args.classifier)
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






    def _load_weights_from_file(self, model, weight_path, flexible = True):
        def _purge(key): # hardcoded logic
            return key.replace("backbone.", "")

        model_dict = model.state_dict()
        weight_dict = torch.load(weight_path)

        if 'model' in weight_dict.keys(): pretrain_dict = weight_dict['model']
        else:   pretrain_dict = weight_dict

        pretrain_dict = { _purge(k) : v for k, v in pretrain_dict.items()}

        if flexible:
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        if not len(pretrain_dict):
            raise Exception(f"No weight names match to be loaded; though file exits ! {weight_path}, Dict: {weight_dict.keys()}")

        lutl.LOG2TXT(f"Pretrained layers:{pretrain_dict.keys()}" ,
                        self.args.gLogPath+"/misc.txt", console=False )

        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

        return model


if __name__ == "__main__":

    from torchsummary import summary

    model = torchvision.models.resnet50()
    summary(model, (3, 224, 224))