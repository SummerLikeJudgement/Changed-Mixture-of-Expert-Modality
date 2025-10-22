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
    def __init__(self, channel, dim_num, t, ratio=8):
        super(router, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )
        self.out = nn.Linear(channel, dim_num, bias=False)
        self.t = t

    def avgpool(self, x):
        b, seq, _ = x.size()
        y = self.avg_pool(x).view(b, seq)
        return y

    def forward(self, x, y, z):
        x = self.avgpool(x)
        y = self.avgpool(y)
        z = self.avgpool(z)
        avg = torch.cat([x, y, z], dim=1)
        output = torch.softmax(
            self.out(
                self.fc(avg)
            )/self.t, dim=1)
        return output



# SE+原始-router网络
# class router(nn.Module):
#     def __init__(self, channel, dim_num, t, dim, ratio=8):
#         super(router, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // ratio, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // ratio, channel, bias=False),
#             nn.Sigmoid()
#         )
#         self.t = t
#         self.l1 = nn.Linear(dim, int(dim/8))
#         self.l2 = nn.Linear(int(dim/8), dim_num)
#
#     def se(self, x):
#         b, seq, _ = x.size()
#         y = self.avg_pool(x).view(b, seq)
#         y = self.fc(y).view(b, seq, 1)
#         return x * y.expand_as(x)
#
#     def forward(self, x, y, z):
#         x = self.se(x)
#         y = self.se(y)
#         z = self.se(z)
#         m = torch.cat((x, y, z), dim=2)
#         m = m.view(m.shape[0], -1)
#         m = (self.l2(
#                     F.relu(
#                         F.normalize(
#                             self.l1(m), p=2, dim=1)))/self.t)
#         output = torch.softmax(m, dim=1)
#         return output