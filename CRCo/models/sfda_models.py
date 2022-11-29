#
# ----------------------------------------------
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from basicda.models import MODELS
import torch.nn.utils.weight_norm as weightNorm


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        print('init batchnorm layer!!!!!!!!!!')
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        print('init linear layer!!!!!!!!!!!!')
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


@MODELS.register_module()
class SFDAResNetBase(nn.Module):
    def __init__(self, resnet_name, bottleneck_dim=256):
        super(SFDAResNetBase, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        #
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim)
        self.bottleneck.apply(init_weights)
        # self.bn.apply(init_weights)
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x, normalize=False):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        # x = F.relu(x)
        if normalize:
            x = F.normalize(x, dim=1)
        return x

    def output_num(self):
        return self.bottleneck_dim

    def optim_parameters(self, lr):
        parameter_list = [{"params": self.feature_layers.parameters(), "lr": lr},
                          {"params": self.bottleneck.parameters(), "lr": lr * 10},
                          {"params": self.bn.parameters(), 'lr': lr * 10},
                          ]
        return parameter_list


@MODELS.register_module()
class SFDAClassifier(nn.Module):
    def __init__(self, num_class, bottleneck_dim=256, type="wn_linear", temp=0.05):
        super(SFDAClassifier, self).__init__()
        self.type = type
        if type == 'wn_linear':
            # self.fc = weightNorm(nn.Linear(bottleneck_dim, num_class,bias=False), name="weight")
            self.fc = weightNorm(nn.Linear(bottleneck_dim, num_class), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, num_class)
            self.fc.apply(init_weights)
        self.temp = temp

    def forward(self, x):
        # x = self.fc(x) / self.temp
        x = self.fc(x)
        return x

    def optim_parameters(self, lr):
        parameter_list = [{"params": self.fc.parameters(), "lr": lr}, ]
        return parameter_list
