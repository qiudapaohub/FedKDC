"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# import sys
# sys.path.append("..")  # 将父文件夹添加到系统路径中
import sys
sys.path.append('../')
# from Semi_decentralized_FD.arguments import args_parser





cfg_cifar = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
cfg_mnist = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M'],
    # 'A' : [64,     'M', 64,      'M', 64,           'M', 64,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x, t):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return F.log_softmax(output, dim=1), F.softmax(output / t, dim=1), output

def make_layers(cfg, args, batch_norm=False):
    layers = []

    input_channel = args.input_channels
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        if l == 256:
            layers += [nn.Dropout2d(p=0.4)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(args):
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        return VGG(make_layers(cfg_mnist['A'], args, batch_norm=True))
    else:
        return VGG(make_layers(cfg_cifar['A'], args, batch_norm=True))

def vgg13_bn(args):
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        return VGG(make_layers(cfg_mnist['B'], args, batch_norm=True))
    else:
        return VGG(make_layers(cfg_cifar['B'], args, batch_norm=True))


def vgg16_bn(args):
    if args.dataset == 'mnist':
        return VGG(make_layers(cfg_mnist['D'], args, batch_norm=True))
    else:
        return VGG(make_layers(cfg_cifar['D'], args, batch_norm=True))

def vgg19_bn(args):
    if args.dataset == 'mnist':
        return VGG(make_layers(cfg_mnist['E'], args, batch_norm=True))
    else:
        return VGG(make_layers(cfg_cifar['E'], args, batch_norm=True))



if __name__ == '__main__':
    args = args_parser()
    # x1 = torch.randn(64, 3, 32, 32)
    x1 = torch.randn(64, 1, 28, 28)
    net1 = vgg11_bn(args)
    # net2 = mobilenet(1, class_num=10)
    a, b, c = net1(x1, t=4)
    print("{} paramerters in total".format(sum(x1.numel() for x1 in net1.parameters())))
    # d, f, g = net2(x2, t=4)
    # print("{} paramerters in total".format(sum(x2.numel() for x2 in net2.parameters())))
    print(a.shape, b.shape, c.shape)
    # print(d.shape, f.shape, g.shape)


