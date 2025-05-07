import torch
import torch.nn as nn

class CNA(nn.Module):
    def __init__(self, channels, factor=8):
        super(CNA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=17)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)

    def forward(self, x_1, x_2):
        b, c, h_1, w = x_1.size()
        _, _, h_2, _ = x_2.size()

        group_x1 = x_1.reshape(b * self.groups, -1, h_1, w)  # b*g,c//g,h,w
        group_x2 = x_2.reshape(b * self.groups, -1, h_2, w)  # b*g,c//g,h,w

        x_h1 = self.pool_h(group_x1) # 1024, 2, 64, 1
        x_w1 = self.pool_w(group_x1).permute(0, 1, 3, 2) # 1024,2,241,1

        x_h2 = self.pool_h(group_x2)
        x_w2 = self.pool_w(group_x2).permute(0, 1, 3, 2)

        hw1 = self.conv1x1(torch.cat([x_h1, x_w1], dim=2))
        x_h1, x_w1 = torch.split(hw1, [h_1, w], dim=2)

        hw2 = self.conv1x1(torch.cat([x_h2, x_w2], dim=2))
        x_h2, x_w2 = torch.split(hw2, [h_2, w], dim=2)

        x1 = self.gn(group_x1 * x_h1.sigmoid() * x_w1.permute(0, 1, 3, 2).sigmoid())
        x2 = self.gn(group_x2 * x_h2.sigmoid() * x_w2.permute(0, 1, 3, 2).sigmoid())

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw

        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw

        tmp = torch.cat([torch.matmul(x11, x12), torch.matmul(x21, x22)], dim=2)
        weights = self.conv1d(tmp)
        weights = weights.reshape(b * self.groups, 1, h_2, w)
        return (group_x2 * weights.sigmoid()).reshape(b, c, h_2, w)