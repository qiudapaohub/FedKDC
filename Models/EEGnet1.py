##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/aliasvishnu/EEGNet
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Feature Extractor """
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class EEGnet1(nn.Module):

    def __init__(self, mtl=True):
        super(EEGnet1, self).__init__()
        self.Conv2d = nn.Conv2d
        self.conv1 = nn.Conv2d(1, 16, (62, 1), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)  # kernel: (2, 2)  stride: (4, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))  # kernel: (2, 4)  stride: (2, 4)

        self.fc1 = nn.Linear(200, 16)
        self.fc2 = nn.Linear(16, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, t):  # x: (pre_batch_size, 7, 400, 62)       (64, 1, 32, 256)     (64, 1, 62, 400)
        x = F.elu(self.conv1(x))  # (pre_batch_size, 16, 400, 1)      (64, 16, 1, 256)     (64, 16, 1, 400)
        x = self.batchnorm1(x)                                      # (64, 16, 1, 256)     (64, 16, 1, 400)
        x = x.permute(0, 2, 1, 3)  # (pre_batch_size, 1, 16, 400)     (64, 1, 16, 256)     (64, 1, 16, 400)
        # Layer 2
        x = self.padding1(x)  # (pre_batch_size, 1, 17, 433)          (64, 1, 17, 289)     (64, 1, 17, 433)
        x = self.dropout(x)
        x = F.elu(self.conv2(x))  # (pre_batch_size, 4, 16, 402)      (64, 4, 16, 258)     (64, 4, 16, 402)
        x = self.batchnorm2(x)                                      # (64, 4, 16, 258)
        x = self.pooling2(x)  # (pre_batch_size, 4, 4, 101)           (64, 4, 4, 65)       (64, 4, 4, 101)
        # Layer 3
        x = self.padding2(x)  # (pre_batch_size, 4, 11, 104)          (64, 4, 11, 68)      (64, 4, 11, 104)
        x = self.dropout(x)
        x = F.elu(self.conv3(x))  # (pre_batch_size, 4, 4, 101)       (64, 4, 4, 65)       (64, 4, 4, 101)
        x = self.batchnorm3(x)                                    # (64, 4, 4, 65)
        x = self.pooling3(x)  # (pre_batch_size, 4, 2, 25)           (64, 4, 2, 16)        (64, 4, 2, 25)
        x = x.reshape(-1, 4*2*25)  # (pre_batch_size, 200)                                 (64, 200)
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1), F.softmax(x / t, dim=1), x

if __name__ == '__main__':
    x1 = torch.randn(64, 1, 62, 400)
    # 实例化模型并打印结构
    model = EEGnet1()
    a, b, c = model(x1, t=4)
    print(a.shape, b.shape, c.shape)

    print(model)