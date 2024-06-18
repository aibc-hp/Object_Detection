# -*- coding: utf-8 -*-
# @Time    : 2024/6/3 9:49
# @Author  : aibc-hp
# @File    : vgg.py
# @Project : YOLOv1
# @Software: PyCharm

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable


__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1470, image_size=448, init_weights=True):
        super(VGG, self).__init__()
        self.features = features  # backbone 的卷积层部分
        self.image_size = image_size
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),  # 参数 inplace=True 表示原地操作，即直接在输入张量上进行相应操作，而不创建新的输出张量
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )  # backbone 的全连接层部分
        if init_weights:
            self._initialize_weights()  # 初始化各网络层的权重和偏置

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.sigmoid(x)  # 归一化到 (0, 1)
        x = x.view(-1, 7, 7, 30)  # 输入为 (-1, 3, 448, 448)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # modules() 是 torch.nn.Module 类的一个内置方法；当调用 self.modules() 时，PyTorch 会递归地遍历模型对象 self 及其所有子对象
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # Xavier 初始化策略；一方面避免梯度消失和爆炸，另一方面加速收敛
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


config = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    first_flag = True
    for v in cfg:
        s = 1
        if v == 64 and first_flag:
            s = 2
            first_flag = False
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)  # *layers 可以把 layers 列表中的元素解包为位置参数


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )


def vgg11(pretrained=False, **kwargs):
    """
    VGG 11-layer model (configuration "A")
    :param pretrained: If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(config['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """
    VGG 11-layer model (configuration "A") with batch normalization
    :param pretrained: If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(config['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """
    VGG 13-layer model (configuration "B")
    :param pretrained: If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(config['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """
    VGG 13-layer model (configuration "B") with batch normalization
    :param pretrained: If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(config['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """
    VGG 16-layer model (configuration "D")
    :param pretrained: If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(config['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """
    VGG 16-layer model (configuration "D") with batch normalization
    :param pretrained: If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(config['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """
    VGG 19-layer model (configuration "E")
    :param pretrained: If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(config['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """
    VGG 19-layer model (configuration 'E') with batch normalization
    :param pretrained: If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(config['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


def test():
    model = vgg16()
    model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1470),
        )
    print(model)
    img = torch.rand(1, 3, 448, 448)
    img = Variable(img)
    output = model(img)
    print(output.size())  # torch.Size([1, 7, 7, 30])


if __name__ == '__main__':
    test()
