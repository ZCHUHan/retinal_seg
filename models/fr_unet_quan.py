import torch
import torch.nn as nn
from .utils import InitWeights_He, get_global_idx, get_next_global_idx
import torch.nn.functional as F
from quantizer.quant_lsq import QuanConv, QuanResize, QuanRELU
from quantizer.act import build_act, Quanhswish
from quantizer.lsq import LsqQuantizer4input
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
        self.conv = QuanConv(in_channels=out_channels, out_channels=out_channels, 
                             kernel_size=kernel_size, padding=padding, bias=use_bias,
                             norm=True)
        self.act = build_act(act_func)
    
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
        self.conv11 = QuanConv(
            in_c, out_c, kernel_size=1, padding=0, bias=False, norm=True)
        self.conv33 = QuanConv(
            in_c, out_c, kernel_size=3, padding=1, bias=False, norm=True)
        self.conv33_di = QuanConv(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2, norm=True)
        
        self.quan_out_1 = LsqQuantizer4input(
                        nbit=8,
                        all_positive=False,
                        per_channel=False
                    ) 
        self.quan_out_2 = LsqQuantizer4input(
                        nbit=8,
                        all_positive=False,
                        per_channel=False
                    )
        self.quan_out_3 = LsqQuantizer4input(
                        nbit=8,
                        all_positive=False,
                        per_channel=False
                    )

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        
        x_1, x_1_scale = self.quan_out_1(x1)
        x_2, x_2_scale = self.quan_out_1(x2)
        x_o, x_o_scale = self.quan_out_2(x_1 + x_2)
        
        x_3, x_3_scale = self.quan_out_2(x3)
        x_o = x_o+x_3
        # x_o, x_o_2_scale = self.quan_out_3(x_o + x_3)
        # xx = x1+x2+x3
        
        # if not self.training and get_global_idx() >= 0: #log npz: 
        #     idx = get_next_global_idx()
        #     np.savez("npz_logging/" + str(idx) + "_add", input1=x_1.detach().cpu().numpy(), input2=x_2.detach().cpu().numpy(), 
        #                 input3=x_2.detach().cpu().numpy(), output=x_o.detach().cpu().numpy())

        return x_o


class up(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(up, self).__init__()
        # conv+bn
        self.up = QuanConv(in_c, out_c, kernel_size=1,norm=True)
        
        # previous version
        self.resize = QuanResize()
        #self.resize = F.interpolate
        
        self.quan_res = LsqQuantizer4input(
                        nbit=8,
                        all_positive=False,
                        per_channel=False
                    ) 
        self.act = build_act("quanhswish")

    def forward(self, x):
        x = self.up(x)
        x = self.act(x)

        #x_r, scale_r = self.quan_res(x)
        x_r = self.resize(x, scale_factor=2, mode='nearest')
        x_r, scale_r = self.quan_res(x_r)
        # if not self.training and get_global_idx() >= 0: #log npz:
        #     idx = get_next_global_idx()
        #     np.savez("npz_logging/" + str(idx) + "_resize", input=x.detach().cpu().numpy(), 
        #              r_scale=scale_r.detach().cpu().numpy(), output= x_r.detach().cpu().numpy())

        return x_r


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        # conv+bn
        self.down = QuanConv(in_c, out_c, kernel_size=2,
                             padding=0, stride=2, bias=False, norm=True)
        self.act = build_act("quanhswish")

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
            self.fuse = QuanConv(in_c, out_c, kernel_size=1, stride=1)

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
        self.final1 = QuanConv(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = QuanConv(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = QuanConv(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = QuanConv(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = QuanConv(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.fuse = QuanConv(
            5, num_classes, kernel_size=1, padding=0, bias=True)
        
        # ? bit
        self.quan_ = nn.ModuleDict({
            name: LsqQuantizer4input(
                bit=8,
                per_channel=False,
                all_positive=False 
            ) 
            for name in ["final1", "final2", "final3", "final4", "final5"]
        })
        
        self.quan_concat = nn.ModuleDict({
                name: LsqQuantizer4input(
                    bit=8,
                    per_channel=False,
                    all_positive=False 
                ) 
                for name in ["cat1_2", "cat2_2_up", "cat1_2_down", "cat2_2", "cat1_1",
                             "cat2_1_up", "cat1_1_down", "cat2_1", "cat3_1_up", "cat2_1_down",
                             "cat3_1", "cat10", "cat20_up", "cat10_down", "cat20", "cat30_up",
                             "cat20_down", "cat30", "cat40_up", "cat11", "cat21_up", 
                             "cat11_down", "cat21", "cat31_up", "cat12", "cat22_up"]
            })
        
        self.apply(InitWeights_He)

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)
        
        qx1_2, r1_2 = self.quan_concat["cat1_2"](x1_2)
        qx_up2_2, r_up2_2 = self.quan_concat["cat2_2_up"](x_up2_2)
        x1_1, x_down1_1 = self.block1_1(torch.cat([qx1_2, qx_up2_2], dim=1))
        
        qx_down1_2, rx_down1_2 = self.quan_concat["cat1_2_down"](x_down1_2)
        qx2_2, rx2_2 = self.quan_concat["cat2_2"](x2_2)
        x2_1, x_up2_1, x_down2_1 = self.block2_1(torch.cat([qx_down1_2, qx2_2], dim=1))
        
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        
        qx1_1, rx1_1 = self.quan_concat["cat1_1"](x1_1)
        qx_up2_1, rx_up2_1  = self.quan_concat["cat2_1_up"](x_up2_1)
        x10, x_down10 = self.block10(torch.cat([qx1_1, qx_up2_1], dim=1))
        
        qx_down1_1, rx_down1_1 = self.quan_concat["cat1_1_down"](x_down1_1)
        qx2_1, rx2_1 = self.quan_concat["cat2_1"](x2_1)
        qx_up3_1, rx_up3_1 = self.quan_concat["cat3_1_up"](x_up3_1)
        x20, x_up20, x_down20 = self.block20(torch.cat([qx_down1_1, qx2_1, qx_up3_1], dim=1))
        
        qx_down2_1, rx_down2_1 = self.quan_concat["cat2_1_down"](x_down2_1)
        qx3_1, rx3_1 = self.quan_concat["cat3_1"](x3_1)
        x30, x_up30 = self.block30(torch.cat([qx_down2_1, qx3_1], dim=1))
        _, x_up40 = self.block40(x_down3_1)
        
        qx10, rx10 = self.quan_concat["cat10"](x10)
        qx_up20, rx_up20 = self.quan_concat["cat20_up"](x_up20)
        x11, x_down11 = self.block11(torch.cat([qx10, qx_up20], dim=1))
        
        qx_down10, rx_down10 = self.quan_concat["cat10_down"](x_down10)
        qx20, rx20 = self.quan_concat["cat20"](x20)
        qx_up30, rx_up30 = self.quan_concat["cat30_up"](x_up30)
        x21, x_up21 = self.block21(torch.cat([qx_down10, qx20, qx_up30], dim=1))
        
        qx_down20, rx_down20 = self.quan_concat["cat20_down"](x_down20)
        qx30, rx30 = self.quan_concat["cat30"](x30)
        qx_up40, rx_up40 = self.quan_concat["cat40_up"](x_up40)
        _, x_up31 = self.block31(torch.cat([qx_down20, qx30, qx_up40], dim=1))
        
        qx11, rx11 = self.quan_concat["cat11"](x11)
        qx_up21, rx_up21 = self.quan_concat["cat21_up"](x_up21)
        x12 = self.block12(torch.cat([qx11, qx_up21], dim=1))
        
        qx_down11, rx_down11 = self.quan_concat["cat11_down"](x_down11) 
        qx21, rx21 = self.quan_concat["cat21"](x21)
        qx_up31, rx_up31 = self.quan_concat["cat31_up"](x_up31)
        _, x_up22 = self.block22(torch.cat([qx_down11, qx21, qx_up31], dim=1))
        
        qx12, rx12 = self.quan_concat["cat12"](x12)
        qx_up22, rx_up22 = self.quan_concat["cat22_up"](x_up22)
        x13 = self.block13(torch.cat([qx12, qx_up22], dim=1))
        
        # if not self.training and get_global_idx() >= 0: #log npz:
        #     idx = get_next_global_idx()
        #     np.savez("npz_logging/" + str(idx)+ "_concat", output=concat_tensor_1.detach().cpu().numpy())
        #     np.savez("npz_logging/" + str(idx)+ "_concat", output=concat_tensor_2.detach().cpu().numpy())
        #     np.savez("npz_logging/" + str(idx)+ "_concat", output=concat_tensor_3.detach().cpu().numpy())
        #     np.savez("npz_logging/" + str(idx)+ "block20" + "_concat", output=concat_tensor_4.detach().cpu().numpy())
        #     np.savez("npz_logging/" + str(idx)+ "block30" + "_concat", output=concat_tensor_5.detach().cpu().numpy())
        #     np.savez("npz_logging/" + str(idx)+ "block11" + "_concat", output=concat_tensor_6.detach().cpu().numpy())
        #     np.savez("npz_logging/" + str(idx)+ "block21" + "_concat", output=concat_tensor_7.detach().cpu().numpy())
        #     np.savez("npz_logging/" + str(idx)+ "block31" + "_concat", output=concat_tensor_8.detach().cpu().numpy())
        #     np.savez("npz_logging/" + str(idx)+ "block12" + "_concat", output=concat_tensor_9.detach().cpu().numpy())
        #     np.savez("npz_logging/" + str(idx)+ "block22" + "_concat", output=concat_tensor_10.detach().cpu().numpy())
        #     np.savez("npz_logging/" + str(idx)+ "block13" + "_concat", output=concat_tensor_11.detach().cpu().numpy())
            
        
        if self.out_ave == True:
            a, r_a = self.quan_["final1"](self.final1(x1_1))
            b, r_b =self.quan_["final1"](self.final2(x10))
            ab, r_ab = self.quan_["final2"](a+b)
            
            c, r_c = self.quan_["final2"](self.final3(x11))
            abc, r_abc = self.quan_["final3"](ab+c)
            
            d, r_d = self.quan_["final3"](self.final4(x12))
            abcd, r_abcd = self.quan_["final4"](abc+d)
            
            e, r_e = self.quan_["final4"](self.final5(x13))
            output, r_out = self.quan_["final5"]((abcd+e)/5)
            
            # output = self.final1(x1_1)+self.final2(x10)+self.final3(x11)+self.final4(x12)+self.final5(x13)/5
            
            # if not self.training and get_global_idx() >= 0: #log npz:
            #     idx = get_next_global_idx()
            #     np.savez("npz_logging/" + str(idx) + "_add", out=tmp.detach().cpu().numpy())
            #     np.savez("npz_logging/" + str(idx) + "_div", out=output.detach().cpu().numpy())
        else:
            output = self.final5(x13)
            

        return output
