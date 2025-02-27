import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')

# input: batchsize*3*500
class dnn(nn.Module):
    def __init__(self, in_num=310, h1=128, h2=64, h3=32, out_num=4):
        super(dnn, self).__init__()
        self.dnn_net = nn.Sequential(
            nn.Linear(in_num, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, out_num)
        )


    def forward(self, x, t):
        out = self.dnn_net(x)
        return F.log_softmax(out, dim=1), F.softmax(out / t, dim=1), out

if __name__ == '__main__':
    x1 = torch.randn(64, 310)
    # 实例化模型并打印结构
    model = dnn()
    print(sum(p.numel() for p in model.parameters()))

