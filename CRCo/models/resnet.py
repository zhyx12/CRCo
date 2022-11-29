from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.autograd import Function
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from basicda.models import MODELS, build_models
from .model_utils import EMA, update_moving_average
from mmcv.runner import get_dist_info
from basicda.utils import concat_all_gather

__all__ = ['ResNet', ]

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    # "resnet50":'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101':
        'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152':
        'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or \
            classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()


def grad_reverse(x, lambd=1.0):
    GradientReverse.scale = lambd
    return GradientReverse.apply(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.nobn = nobn

    def forward(self, x, source=True):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        print(self.scale)
        return input * self.scale


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride
        self.nobn = nobn

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


@MODELS.register_module(name='resnet')
class ResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, num_classes=1000, pretrained=True, set_bn_weight_decay_zero=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        #
        assert depth in [18, 34, 50, 101, 152], 'wrong depth for resnet'
        block = self.arch_settings[depth][0]
        layers = self.arch_settings[depth][1]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in1 = nn.InstanceNorm2d(64)
        self.in2 = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #
        if pretrained:
            pretrained_dict = model_zoo.load_url(model_urls['resnet{}'.format(depth)])
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        #
        self.set_bn_weight_decay_zero = set_bn_weight_decay_zero

    def _make_layer(self, block, planes, blocks, stride=1, nobn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nobn=nobn))
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    optim_param.append({'params': param, 'lr': lr * 10})
                    print('{} will be optimized, lr {}'.format(name, lr * 10))
                else:
                    if self.set_bn_weight_decay_zero:
                        if 'bn' in name:
                            optim_param.append({'params': param, 'lr': lr, 'weight_decay': 0.0})
                            print('{} will be optimized, lr {}, zero decay'.format(name, lr))
                    else:
                        optim_param.append({'params': param, 'lr': lr})
                        print('{} will be optimized, lr {}'.format(name, lr))
            else:
                print('{} will be ignored'.format(name))

        return optim_param

@MODELS.register_module(name='resnet_with_norm')
class ResNetWithNorm(ResNet):
    def __init__(self, depth, num_classes=1000, pretrained=True, set_bn_weight_decay_zero=False, fc1_dim=512):
        super(ResNetWithNorm, self).__init__(depth=depth, num_classes=num_classes, pretrained=pretrained,
                                            set_bn_weight_decay_zero=set_bn_weight_decay_zero)
        #

    def forward(self, x, normalize=True):
        x = super(ResNetWithNorm, self).forward(x)
        if normalize:
            x = F.normalize(x)
        return x

@MODELS.register_module(name='resnet_with_fc1')
class ResNetWithFC1(ResNet):
    def __init__(self, depth, num_classes=1000, pretrained=True, set_bn_weight_decay_zero=False, fc1_dim=512):
        super(ResNetWithFC1, self).__init__(depth=depth, num_classes=num_classes, pretrained=pretrained,
                                            set_bn_weight_decay_zero=set_bn_weight_decay_zero)
        #
        block = self.arch_settings[depth][0]
        self.classifier_fc1 = nn.Linear(512 * block.expansion, fc1_dim)

    def forward(self, x, normalize=True):
        x = super(ResNetWithFC1, self).forward(x)
        x = self.classifier_fc1(x)
        if normalize:
            x = F.normalize(x)
        return x


@MODELS.register_module(name='ema_model')
class EMAModel(nn.Module):
    def __init__(self, model_dict, moving_average_decay=0.99):
        super(EMAModel, self).__init__()
        self.online_network = build_models(model_dict)
        self.target_network = build_models(model_dict)
        self.target_ema_updater = EMA(moving_average_decay)
        #
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    # TODO: 模型输入需要考虑多个输入的情况
    def forward(self, x1, x2=None, **kwargs):
        x1 = self.online_network(x1, **kwargs)
        if x2 is not None:
            with torch.no_grad():
                self.update_moving_average()
                x2 = self.target_network(x2, **kwargs)
            return x1, x2
        else:
            return x1

    def update_moving_average(self):
        update_moving_average(self.target_ema_updater, self.target_network, self.online_network)

    def optim_parameters(self, lr):
        if hasattr(self.online_network, 'optim_parameters'):
            return self.online_network.optim_parameters(lr)
        else:
            return [param for param in self.online_network.parameters()]
