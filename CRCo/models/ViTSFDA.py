import torch
import torch.nn as nn
import torch.nn.functional as F

from .ViT import VT, vit_model
# from .grl import WarmStartGradientReverseLayer
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


@MODELS.register_module(name='vit_sfda_basenet')
class ViTSFDABaseNet(nn.Module):
    def __init__(self, base_net='vit_base_patch16_224', use_bottleneck=True, bottleneck_dim=1024, width=1024,
                 sfda_feat_width=256, args=None):
        super(ViTSFDABaseNet, self).__init__()

        self.base_network = vit_model[base_net](pretrained=True, args=args, VisionTransformerModule=VT)
        self.use_bottleneck = use_bottleneck
        # self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=True)
        if self.use_bottleneck:
            self.bottleneck_layer = [nn.Linear(self.base_network.embed_dim, bottleneck_dim),
                                     nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
            self.bottleneck = nn.Sequential(*self.bottleneck_layer)

        classifier_dim = bottleneck_dim if use_bottleneck else self.base_network.embed_dim
        self.classifier_layer = [nn.Linear(classifier_dim, width), nn.ReLU(), nn.Dropout(0.5)]
        self.classifier = nn.Sequential(*self.classifier_layer)
        #
        if self.use_bottleneck:
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
        #
        self.classifier[0].weight.data.normal_(0, 0.01)
        self.classifier[0].bias.data.fill_(0.0)
        #
        self.sfda_bottleneck = nn.Linear(width, sfda_feat_width)
        self.sfda_bn = nn.BatchNorm1d(sfda_feat_width)
        self.sfda_bottleneck.apply(init_weights)

    def forward(self, inputs):
        features = self.base_network.forward_features(inputs)
        if self.use_bottleneck:
            features = self.bottleneck(features)
        feat = self.classifier(features)
        feat = self.sfda_bottleneck(feat)
        feat = self.sfda_bn(feat)
        return feat



    def optim_parameters(self, lr):
        optim_param = [
            {"params": self.base_network.parameters(), "lr": lr * 0.1},
            {"params": self.classifier.parameters(), "lr": lr * 1.0},
            {"params": self.sfda_bottleneck.parameters(), "lr": lr * 10.0},
            {"params": self.sfda_bn.parameters(), "lr": lr * 10.0},
        ]
        #
        if self.use_bottleneck:
            optim_param.extend([{"params": self.bottleneck.parameters(), "lr": lr * 1.0}])

        return optim_param


@MODELS.register_module(name='vit_sfda_classifier')
class VitSFDAClassifier(nn.Module):
    def __init__(self, width=256, class_num=65,type="wn_linear", ):
        super(VitSFDAClassifier, self).__init__()

        if type == 'wn_linear':
            self.fc = weightNorm(nn.Linear(width, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(width, class_num)
            self.fc.apply(init_weights)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

    def forward(self, inputs):
        return self.fc(inputs)

    def optim_parameters(self, lr):
        optim_param = [
            {"params": self.fc.parameters(), "lr": lr * 1.0}, ]

        return optim_param
