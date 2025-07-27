from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale  # 计算模板图像和搜索图像的交叉相关
    
    def _fast_xcorr(self, z, x):  # 快速的二维交叉相关操作
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)  # 将搜索图像 x 重新调整形状以匹配模板图像 z 的大小
        out = F.conv2d(x, z, groups=nz)  # groups=nz 表示每个通道的输入 x 与对应的模板图像 z 进行单独的卷积
        out = out.view(nx, -1, out.size(-2), out.size(-1))  # 输出重新调整形状以匹配输入搜索图像的数量 nx
        return out
