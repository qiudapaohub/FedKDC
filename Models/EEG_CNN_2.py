import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')

# input: batchsize*3*500
class DeprNet(nn.Module):
    def __init__(self, num_classes):
        super(DeprNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=0, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=(1, 1), padding=(1, 2), bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=(1, 1), padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32*76, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(128, num_classes, bias=True)


    def forward(self, x, t):
        out = self.layer1(x)       # (64, 32, 31, 200)    (64, 64, 113, 4)   (3000, 32, 76)
        # out = self.layer2(out)     # (64, 64, 15, 100)    (64, 64, 56, 4)

        # out = self.layer3(out)     # (64, 128, 15, 100)   (64, 128, 56, 4)
        # out = self.layer4(out)     # (64, 128, 7, 50)     (64, 128, 28, 3)
        # out = self.layer5(out)     # (64, 256, 7, 50)     (64, 256, 28, 3)
        # out = self.layer6(out)     # (64, 256, 3, 25)     (64, 256, 14, 2)
        out = out.reshape(out.size(0), -1)  # (64, 19200)  (64, 7168)
        out = self.fc1(out)        # (64, 44800)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1), F.softmax(out / t, dim=1), out



if __name__ == '__main__':
    x1 = torch.randn(3000, 1, 310)
    # 实例化模型并打印结构
    model = DeprNet(3)
    print(sum(p.numel() for p in model.parameters()))
    model2 = DeprNet(4)
    print(sum(p.numel() for p in model2.parameters()))
