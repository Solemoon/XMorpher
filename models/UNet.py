import sys

sys.path.append('..')

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.STN import SpatialTransformer, Re_SpatialTransformer
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//4, out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//4, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet_base(nn.Module):
    def __init__(self, n_channels, chs=(16, 32, 64, 128, 256, 128, 64, 32, 16)):
        super(UNet_base, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, chs[0])
        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])
        self.up1 = Up(chs[4] + chs[3], chs[5])
        self.up2 = Up(chs[5] + chs[2], chs[6])
        self.up3 = Up(chs[6] + chs[1], chs[7])
        self.up4 = Up(chs[7] + chs[0], chs[8])
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)

    def forward(self, x):
        Z = x.size()[2]
        Y = x.size()[3]
        X = x.size()[4]
        diffZ = (16 - Z % 16) % 16
        diffY = (16 - Y % 16) % 16
        diffX = (16 - X % 16) % 16
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x[:, :, diffZ//2: Z+diffZ//2, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2]

class UNet_reg(nn.Module):
    def __init__(self, n_channels=1, depth=(16, 32, 64, 128, 256, 128, 64, 32, 16)):
        super(UNet_reg, self).__init__()
        self.unet = UNet_base(n_channels=n_channels*2, chs=depth)
        self.out_conv = nn.Conv3d(depth[-1], 3, 1)
        self.stn = SpatialTransformer()
        self.rstn = Re_SpatialTransformer()
    def forward(self, moving, fixed, mov_label=None, fix_label=None):
    # def forward(self, moving, fixed, mov_label=None, fix_label=None, raw=None, grid = None):
        x = torch.cat([fixed, moving], dim=1)
        x = self.unet(x)
        flow = self.out_conv(x)
        # return flow
        w_m_to_f = self.stn(moving, flow)
        # w_m_to_f = self.stn(raw, flow)
        # grid = self.stn(grid, flow, mode='nearest')

        w_f_to_m = self.rstn(fixed, flow)

        if mov_label is not None:
            w_label_m_to_f = self.stn(mov_label, flow, mode='nearest')
        else:
            w_label_m_to_f = None

        if fix_label is not None:
            w_label_f_to_m = self.rstn(fix_label, flow, mode='nearest')
        else:
            w_label_f_to_m = None

        return w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow

class UNet_app(nn.Module):
    def __init__(self, n_channels=1, depth=(16, 32, 64, 128, 256, 128, 64, 32, 16)):
        super(UNet_app, self).__init__()
        self.unet = UNet_base(n_channels=n_channels * 2, chs=depth)
        self.out_conv = nn.Conv3d(depth[-1], 1, 1)

    def forward(self, atlas, fixed):
        x = torch.cat([fixed, atlas], dim=1)
        x = self.unet(x)
        app = self.out_conv(x)
        out = app + atlas

        return out, app

class UNet_seg(nn.Module):
    def __init__(self, n_channels=1, n_classes=8, depth=(16, 32, 64, 128, 256, 128, 64, 32, 16)):
        super(UNet_seg, self).__init__()
        self.unet = UNet_base(n_channels=n_channels, chs=depth)
        self.out_conv = nn.Conv3d(depth[-1], n_classes, 1)
    def forward(self, x):
        x = self.unet(x)
        x = self.out_conv(x)
        return x

if __name__ == "__main__":
    print("begin test:")
    reg = UNet_reg()
    input1 = torch.randn(size=(1,1,144,144,144)) # (B, C, D, H, W)
    input2 = torch.randn(size=(1, 1, 144, 144, 144))  # (B, C, D, H, W)
    output = reg(input1,input2)
    print(output.shape)