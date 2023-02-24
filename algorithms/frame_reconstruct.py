import os, sys
import torch
import torchvision
import torch.nn as nn

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl

""" UNUSED FOR NOW """

class ResNetAutoEncoder(nn.Module):

    def __init__(self, configs, bottleneck):

        super(ResNetAutoEncoder, self).__init__()

        self.backbone, self.latent_size = self._load_resnet_coders()
        self.decoder = ResNetDecoder(configs=configs[::-1], bottleneck=bottleneck)

    def forward(self, x):

        x = self.backbone(x)
        x = self.decoder(x)

        return x


    def _load_resnet_backbone(self):

        torch_pretrain = "DEFAULT" if self.args.featx_pretrain == "DEFAULT" else None

        if self.args.feature_extract == 'resnet18':
            backbone = torchvision.models.resnet18(zero_init_residual=True,
                                 weights=torch_pretrain)
            decoder =  ResNetDecoder([2, 2, 2, 2], bottleneck=False)
            latent_size = 512
        elif self.args.feature_extract == 'resnet50':
            backbone = torchvision.models.resnet50(zero_init_residual=True,
                                weights=torch_pretrain)
            decoder =  ResNetDecoder([3, 4, 6, 3], bottleneck=True)
            latent_size = 2048

        backbone.fc = nn.Identity() #remove fc of default arch

        # Change input Conv
        # conv1_weight = torch.sum(backbone.conv1.weight, axis = 1).unsqueeze(1)
        # backbone.conv1 = nn.Conv2d(1,  64, kernel_size=7, stride=2, padding=3, bias=False)
        # with torch.no_grad():
        #     backbone.conv1.weight.copy_(conv1_weight)

        # pretrain from external file
        if os.path.exists(self.args.featx_pretrain):
            backbone = self._load_weights_from_file(backbone,
                                self.args.featx_pretrain )
            print("Loaded:", self.args.featx_pretrain )

        # freeze model
        if self.args.featx_freeze:
            print("Freezing Resnet weights ...")
            for param in backbone.parameters():
                param.requires_grad = False

        return backbone, outfeat_size




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



##================== ResNet Encoder Decoder ====================================
## Refer: https://github.com/Horizon2333/imagenet-autoencoder


def get_configs(arch='resnet50'):

    # True or False means wether to use BottleNeck

    if arch == 'resnet18':
        return [2, 2, 2, 2], False
    elif arch == 'resnet34':
        return [3, 4, 6, 3], False
    elif arch == 'resnet50':
        return [3, 4, 6, 3], True
    elif arch == 'resnet101':
        return [3, 4, 23, 3], True
    elif arch == 'resnet152':
        return [3, 8, 36, 3], True
    else:
        raise ValueError("Undefined model")


class ResNet(nn.Module):

    def __init__(self, configs, bottleneck=False, num_classes=1000):
        super(ResNet, self).__init__()

        self.encoder = ResNetEncoder(configs, bottleneck)

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        if bottleneck:
            self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        else:
            self.fc = nn.Linear(in_features=512, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.encoder(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class ResNetEncoder(nn.Module):

    def __init__(self, configs, bottleneck=False):
        super(ResNetEncoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:

            self.conv2 = EncoderBottleneckBlock(in_channels=64,   hidden_channels=64,  up_channels=256,  layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderBottleneckBlock(in_channels=256,  hidden_channels=128, up_channels=512,  layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderBottleneckBlock(in_channels=512,  hidden_channels=256, up_channels=1024, layers=configs[2], downsample_method="conv")
            self.conv5 = EncoderBottleneckBlock(in_channels=1024, hidden_channels=512, up_channels=2048, layers=configs[3], downsample_method="conv")

        else:

            self.conv2 = EncoderResidualBlock(in_channels=64,  hidden_channels=64,  layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderResidualBlock(in_channels=64,  hidden_channels=128, layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderResidualBlock(in_channels=128, hidden_channels=256, layers=configs[2], downsample_method="conv")
            self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512, layers=configs[3], downsample_method="conv")

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x



class ResNetDecoder(nn.Module):

    def __init__(self, configs, bottleneck=False):
        super(ResNetDecoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:

            self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024, layers=configs[0])
            self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,  layers=configs[1])
            self.conv3 = DecoderBottleneckBlock(in_channels=512,  hidden_channels=128, down_channels=256,  layers=configs[2])
            self.conv4 = DecoderBottleneckBlock(in_channels=256,  hidden_channels=64,  down_channels=64,   layers=configs[3])


        else:

            self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[0])
            self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=configs[1])
            self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=configs[2])
            self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

class EncoderResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, layers, downsample_method="conv"):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=True)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False)

                self.add_module('%02d EncoderLayer' % i, layer)

        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=False)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False)

                self.add_module('%02d EncoderLayer' % (i+1), layer)

    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, layers, downsample_method="conv"):
        super(EncoderBottleneckBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=True)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)

                self.add_module('%02d EncoderLayer' % i, layer)

        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)

                self.add_module('%02d EncoderLayer' % (i+1), layer)

    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False)

            self.add_module('%02d EncoderLayer' % i, layer)

    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=in_channels, upsample=False)

            self.add_module('%02d EncoderLayer' % i, layer)


    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class EncoderResidualLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, downsample):
        super(EncoderResidualLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x

class EncoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, downsample):
        super(EncoderBottleneckLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif (in_channels != up_channels):
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        x = self.relu(x)

        return x

class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        else:
            self.upsample = None

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x

class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.upsample = None
            self.down_scale = None

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x



if __name__ == "__main__":
    from torchsummary import summary

    model = torchvision.models.resnet50()
    model.fc = nn.Identity()
    summary(model, (3, 224, 224), depth=3)

    configs, bottleneck_f = get_configs("resnet152")
    encoder = ResNetEncoder(configs, bottleneck_f)
    decoder = ResNetDecoder(configs[::-1], bottleneck_f)


    input = torch.randn((5,3,224,224))
    print(input.shape)
    output = encoder(input)
    print(output.shape)
    output = decoder(output)
    print(output.shape)

    summary(encoder, (3, 224, 224), depth=3)