import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv1d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv1d_1x1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv1d_1x1(y)
        return y


class EEGNet(nn.Module):
    def __init__(self, nb_classes: int, Chans: int = 32, Samples: int = 256,
                 dropoutRate: float = 0.5, kernLength: int = 63,
                 F1:int = 8, D:int = 2):
        super().__init__()

        F2 = F1 * D

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        # In: (B, Chans, Samples, 1)
        # Out: (B, F1, Samples, 1)
        self.conv1 = nn.Conv1d(Chans, F1, kernLength, padding=(kernLength // 2))
        self.bn1 = nn.BatchNorm1d(F1) # (B, F1, Samples, 1)
        # In: (B, F1, Samples, 1)
        # Out: (B, F2, Samples - Chans + 1, 1)
        self.conv2 = nn.Conv1d(F1, F2, Chans, groups=F1)
        self.bn2 = nn.BatchNorm1d(F2) # (B, F2, Samples - Chans + 1, 1)
        # In: (B, F2, Samples - Chans + 1, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.avg_pool = nn.AvgPool1d(4)
        self.dropout = nn.Dropout(dropoutRate)

        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.conv3 = SeparableConv1d(F2, F2, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(F2)
        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 32, 1)
        self.avg_pool2 = nn.AvgPool1d(8)
        # In: (B, F2 *  (Samples - Chans + 1) / 32)
        self.fc = nn.Linear(F2 * ((Samples - Chans + 1) // 32), nb_classes)

    def forward(self, x: torch.Tensor, t):
        # Block 1 (1 64 128)   (1 3 500)    (64,32,256)
        y1 = self.conv1(x)    # (1 8 128)  (1 8 500)  (64,8,256)
        #print("conv1: ", y1.shape)
        y1 = self.bn1(y1)    # (1 8 128)   (1 8 500)  (64,8,256)

        y1 = self.dropout(y1)
        #print("bn1: ", y1.shape)
        y1 = self.conv2(y1)  # (1 16 65)   (1 16 498)  (64,16,255)
        #print("conv2", y1.shape)
        y1 = F.relu(self.bn2(y1))  # (1 16 65)   (1 16 498)  (64,16,255)
        #print("bn2", y1.shape)
        y1 = self.avg_pool(y1)   # (1 16 16)   (1 16 124)   (64,16,56)
        #print("avg_pool", y1.shape)
        y1 = self.dropout(y1)     # (1 16 16)  (1 16 124)   (64,16,56)
        #print("dropout", y1.shape)

        # Block 2
        y2 = self.conv3(y1)      # (1 16 16)   (1 16 124)   (64,16,56)
        #print("conv3", y2.shape)
        y2 = F.relu(self.bn3(y2)) # (1 16 16)  (1 16 124)   (64,16,56)
        #print("bn3", y2.shape)
        y2 = self.avg_pool2(y2)   # (1 16 2)   (1 16 15)    (64,16,7)
        #print("avg_pool2", y2.shape)
        y2 = self.dropout(y2)    # (1 16 2)    (1 16 15)    (64,16,7)
        #print("dropout", y2.shape)
        y2 = torch.flatten(y2, 1)  # (1 32)    (1 240)      (64,112)
        #print("flatten", y2.shape)
        y2 = self.fc(y2)     # (1 4)    (64,3)
        #print("fc", y2.shape)

        return F.log_softmax(y2, dim=1), F.softmax(y2 / t, dim=1), y2


if __name__ == '__main__':
    x1 = torch.randn(64, 32, 256)
    # 实例化模型并打印结构
    # EEGNet hyperparams
    NB_CLASSES = 3
    KERNEL_LENGTH = 63
    CHANNELS = 64
    SAMPLES = 128
    F1 = 8
    D = 2

    model = EEGNet(NB_CLASSES, kernLength=KERNEL_LENGTH)
    a = model(x1, t=1)
    print(a.shape)

    print(model)