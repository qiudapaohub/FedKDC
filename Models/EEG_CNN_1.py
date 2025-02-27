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
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64*36, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(256, num_classes, bias=True)


    def forward(self, x, t):
        out = self.layer1(x)       # (64, 32, 31, 200)   (64, 64, 113, 4)  (3000, 32, 76)
        out = self.layer2(out)     # (64, 64, 15, 100)   (64, 64, 56, 4)   (3000, 64, 36)

        # out = self.layer3(out)     # (64, 128, 7, 50)    (64, 128, 28, 2)   (3000, 128, 17)
        out = out.reshape(out.size(0), -1)  # (64, 44800)    (64, 7168)     (3000, 2176)
        out = self.fc1(out)        # (64, 44800)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1), F.softmax(out / t, dim=1), out
    # -----------------------------------------------------------------------------------
    # def __init__(self):
    #     super(DeprNet, self).__init__()
    #     # 第一层卷积
    #     self.conv1 = nn.Conv1d(62, 128, kernel_size=5, stride=2, padding=0)
    #     self.bn1 = nn.BatchNorm1d(128)
    #     self.pool1 = nn.MaxPool1d(2, stride=2)
    #
    #     # 第二层卷积
    #     self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=0)
    #     self.bn2 = nn.BatchNorm1d(64)
    #     self.pool2 = nn.MaxPool1d(2, stride=2)
    #
    #     # 第三层卷积
    #     self.conv3 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0)
    #     self.bn3 = nn.BatchNorm1d(64)
    #     self.pool3 = nn.MaxPool1d(2, stride=2)
    #
    #     # 第四层卷积
    #     self.conv4 = nn.Conv1d(64, 32, kernel_size=2, stride=1, padding=0)
    #     self.bn4 = nn.BatchNorm1d(32)
    #     self.pool4 = nn.MaxPool1d(2, stride=2)
    #
    #     # 第五层卷积
    #     # self.conv5 = nn.Conv1d(64, 32, kernel_size=2, stride=1, padding=0)
    #     # self.bn5 = nn.BatchNorm1d(32)
    #     # self.pool5 = nn.MaxPool1d(2, stride=2)
    #
    #     # 全连接层
    #     self.fc1 = nn.Linear(320, 16)  # 注意这里的 30 需要根据实际输出尺寸调整
    #     self.fc2 = nn.Linear(16, 3)
    #
    #     self.dropout1 = nn.Dropout(0.2)
    #     self.dropout2 = nn.Dropout(0.3)
    #     self.dropout3 = nn.Dropout(0.5)
    #
    # def forward(self, x, t):
    #     x = F.relu(self.bn1(self.conv1(x)))  # (bs, 128, 126)     (bs, 128, 198)   (bs, 128, 398)
    #     x = self.pool1(x)                    # (bs, 128, 63)      (bs, 128, 99)    (bs, 128, 199)
    #     x = self.dropout1(x)
    #     x = F.relu(self.bn2(self.conv2(x)))  # (bs, 64, 30)       (bs, 64, 49)     (bs, 64, 99)
    #     x = self.dropout2(x)
    #     x = self.pool2(x)                    # (bs, 64, 15)       (bs, 64, 24)     (bs, 64, 49)
    #     x = F.relu(self.bn3(self.conv3(x)))  # (bs, 64, 11)        (bs, 64, 22)    (bs, 64, 47)
    #     x = self.dropout2(x)
    #     x = self.pool3(x)                    # (bs, 64, 5)        (bs, 64, 11)     (bs, 64, 23)
    #     x = F.relu(self.bn4(self.conv4(x)))  # (bs, 32, 4)        (bs, 32, 10)     (bs, 32, 22)
    #     x = self.pool4(x)                    # (bs, 32, 2)         (bs, 32, 5)     (bs, 32, 11)
    #
    #     # x = F.relu(self.bn5(self.conv5(x)))  # (bs, 32, 121)
    #     # x = self.pool5(x)                    # (bs, 32, 60)
    #     x = x.view(x.size(0), -1)  # 扁平化   # (bs, 64)   (64, 8064)   (64, 160)
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout3(x)
    #     x = self.fc2(x)
    #     return F.log_softmax(x, dim=1), F.softmax(x / t, dim=1), x


if __name__ == '__main__':
    x1 = torch.randn(3000, 1, 310)
    # 实例化模型并打印结构
    model = DeprNet(3)
    print(sum(p.numel() for p in model.parameters()))
    model2 = DeprNet(4)
    print(sum(p.numel() for p in model2.parameters()))
    