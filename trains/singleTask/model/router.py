import torch
import torch.nn as nn
import torch.nn.functional as F

# 原始-router网络
# class router(nn.Module):
#     def __init__(self, dim, channel_num, t):
#         super().__init__()
#         self.l1 = nn.Linear(dim, int(dim/8))
#         self.l2 = nn.Linear(int(dim/8), channel_num)
#         self.t = t # 温度用于控制权重大小
#
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = (self.l2(
#             F.relu(
#                 F.normalize(
#                     self.l1(x), p=2, dim=1)))/self.t)
#         output = torch.softmax(x, dim=1)
#         return output


# SE-router网络
class router(nn.Module):
    def __init__(self, channel, dim_num, ratio):
        super(router, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel * ratio), channel, bias=False),
            nn.Sigmoid()
        )
        self.out = nn.Linear(channel, dim_num, bias=False)

    def forward(self, x):
        b, seq ,_ = x.size()
        y = self.avg_pool(x).view(b, seq)
        y = self.fc(y)
        y = self.out(y)
        output = torch.softmax(y, dim=1)
        return output