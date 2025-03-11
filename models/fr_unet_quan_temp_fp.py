import torch
import torch.nn as nn
from .utils import InitWeights_He
import torch.nn.functional as F
import numpy as np

class conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        padding=1,
        use_bias=False,
        dropout_rate=0,
        norm=True,
        act_func="quanhswish",
    ):
        super(conv, self).__init__()

        self.dropout = nn.Dropout2d(dropout_rate, inplace=False) if dropout_rate > 0 else None
        # conv+bn
        
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding) #, bias=False
        self.act = nn.Hardswish()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.act(x)
        
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.act(x)
                
        return x


class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_c, out_c, kernel_size=1, padding=0) #, bias=False
        self.conv33 = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=1) #, bias=False
        self.conv33_di = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=2, dilation=2) #, bias=False
        

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        
        xx = x1+x2+x3

        return xx


class up(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(up, self).__init__()

        # conv+bn
        self.up = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.resize = F.interpolate
        self.act = nn.Hardswish()
        

    def forward(self, x):
        x = self.up(x)
        x = self.act(x)
        x_r = self.resize(x, scale_factor=2, mode='bilinear')
        
        return x_r


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        # conv+bn
        self.down = nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2) #, bias=False
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.down(x)
        x = self.act(x)
        return x


class block(nn.Module):
    def __init__(self, in_c, out_c,  dp=0, is_up=False, is_down=False, fuse=False):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = conv(out_c, out_c, dropout_rate=dp)
        if self.is_up == True:
            self.up = up(out_c, out_c//2)
        if self.is_down == True:
            self.down = down(out_c, out_c*2)

    def forward(self,  x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


class FR_UNet_Quan(nn.Module):
    def __init__(self,  num_classes=1, num_channels=1, feature_scale=2,  dropout=0.2, fuse=True, out_ave=True, upsample="BiLinear"):
        super(FR_UNet_Quan, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block12 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block13 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block22 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block31 = block(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block40 = block(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.fuse = nn.Conv2d(
            5, num_classes, kernel_size=1, padding=0, bias=True)
        
        self.apply(InitWeights_He)

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)
        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        _, x_up40 = self.block40(x_down3_1)
        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))
        if self.out_ave == True:
            output = (self.final1(x1_1)+self.final2(x10) +
                      self.final3(x11)+self.final4(x12)+self.final5(x13))/5
        else:
            output = self.final5(x13)

        return output