import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = [
    'cnn_mnist', 'cnn_cifar10'
]

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, t):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), F.softmax(x / t, dim=1), x

# class Net(nn.Module):
#     def __init__(self, model_name):
#         super(Net, self).__init__()
#         self.model_name = model_name
#         if self.model_name == 'mnist':
#             self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#             self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#             self.fc1 = nn.Linear(, 100)
#             self.fc2 = nn.Linear(100, 10)
#         elif self.model_name == 'cifar10':
#             self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#             self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#             self.fc1 = nn.Linear(20, 100)
#             self.fc2 = nn.Linear(100, 10)
#         else:
#             self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#             self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#             self.fc1 = nn.Linear(20, 200)
#             self.fc2 = nn.Linear(200, 100)
#         # self.conv2_drop = nn.Dropout2d()
#
#     def forward(self, x, t):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = x.view(x.size()[0], -1)
#
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1), F.softmax(x / t, dim=1), x


class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, t):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), F.softmax(x / t, dim=1), x

def cnn_mnist():
    model = CNNMnist()
    return model

def cnn_cifar10():
    model = CNNCifar10()
    return model


if __name__ == '__main__':
    x1 = torch.randn(64, 3, 32, 32)
    x2 = torch.randn(64, 1, 28, 28)
    net1 = cnn_cifar10()
    net2 = cnn_mnist()
    a, b, c = net1(x1, t=4)
    d, f, g = net2(x2, t=4)

    print(a.shape, b.shape, c.shape)
    print(d.shape, f.shape, g.shape)
