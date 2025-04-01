import torch
import torch.nn as nn
from .utils import InitWeights_He
import torch.nn.functional as F
from quantizer.quant_lsq_double import QuanConv, QuanResize, QuanRELU, get_global_idx, get_next_global_idx, SymmetricQuantFunction
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
        quan_input = True
    ):
        super(conv, self).__init__()

        self.dropout = nn.Dropout2d(dropout_rate, inplace=False) if dropout_rate > 0 else None
        # conv+bn
        self.conv = QuanConv(in_channels=out_channels, out_channels=out_channels, 
                             kernel_size=kernel_size, padding=padding, bias=use_bias,
                             norm=True, quan_input = quan_input)
        self.act = build_act(act_func)
        self.conv2 = QuanConv(in_channels=out_channels, out_channels=out_channels, 
                             kernel_size=kernel_size, padding=padding, bias=use_bias,
                             norm=True)
        self.act2 = build_act(act_func)
    
    def forward(self, x: torch.Tensor, scale_x=None) -> torch.Tensor:
        x = self.conv(x, scale_x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.act(x)
        
        x = self.conv2(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.act2(x)
                
        return x


class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c, quan_input = True):
        super(feature_fuse, self).__init__()
        self.conv11 = QuanConv(
            in_c, out_c, kernel_size=1, padding=0, bias=False, norm=True, quan_input = False)
        self.conv33 = QuanConv(
            in_c, out_c, kernel_size=3, padding=1, bias=False, norm=True, quan_input = False)
        self.conv33_di = QuanConv(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2, norm=True, quan_input = False)
        
        self.quan_input = quan_input

        self.quan_ = nn.ModuleDict({
                    "fuse_1_16bit": LsqQuantizer4input(bit=16, per_channel=False, all_positive=False),
                    "fuse_2_16bit": LsqQuantizer4input(bit=16, per_channel=False, all_positive=False),
                    "fuse_1_8bit": LsqQuantizer4input(bit=8, per_channel=False, all_positive=False),
                    "fuse_2_8bit": LsqQuantizer4input(bit=8, per_channel=False, all_positive=False),
                    "fuse_3_8bit": LsqQuantizer4input(bit=8, per_channel=False, all_positive=False)
                })
        
        if(self.quan_input):
            self.quan_["input_8_bit"] = LsqQuantizer4input(bit=8, per_channel=False, all_positive=False)
        

    def forward(self, x, scale_x=None):

        if(self.quan_input):
            x, scale_x = self.quan_["input_8_bit"](x)

        x1 = self.conv11(x, scale_x)
        x2 = self.conv33(x, scale_x)
        x3 = self.conv33_di(x, scale_x)
        
        # x_1, x_1_scale = self.quan_out_1(x1)
        # x_2, x_2_scale = self.quan_out_1(x2)
        # x_o, x_o_scale = self.quan_out_2(x_1 + x_2)
        
        # x_3, x_3_scale = self.quan_out_2(x3)
        # x_o = x_o+x_3
        
        a, r_a = self.quan_["fuse_1_16bit"](x1) # 8, 16
        b, r_b = self.quan_["fuse_1_8bit"](x2)
        #b, _ = self.quan_["fuse_1_16bit"](b) # No need to unifi add node output scale
        
        ab, r_ab = self.quan_["fuse_2_8bit"](a+b)    # 8, 16
        if not self.training and get_global_idx() >= 0: #log npz: 
            idx = get_next_global_idx()
            np.savez("npz_logging/" + str(idx) + "_add", a_scale=r_a.detach().cpu().numpy(),
                                                         b_scale=r_b.detach().cpu().numpy(),
                                                         a_bitdepth = 16,
                                                         b_bitdepth = 8,
                                                         a = a.detach().cpu().numpy(),
                                                         b = b.detach().cpu().numpy())
        
        c, r_c = self.quan_["fuse_2_16bit"](x3)
        #ab, _ = self.quan_["fuse_2_16bit"](ab) # Add Node output can only be 8-bit and already quantized in line 80
        #abc, r_abc = self.quan_["fuse_3_8bit"](ab+c) # 8 # output quan handle by the next conv node
        
        # x_o, x_o_2_scale = self.quan_out_3(x_o + x_3)
        # xx = x1+x2+x3
        
        if not self.training and get_global_idx() >= 0: #log npz: 
            idx = get_next_global_idx()
            np.savez("npz_logging/" + str(idx) + "_add", a_scale=r_ab.detach().cpu().numpy(),
                                                         b_scale=r_c.detach().cpu().numpy(),
                                                         a_bitdepth = 8,
                                                         b_bitdepth = 16,
                                                         a = ab.detach().cpu().numpy(),
                                                         b = c.detach().cpu().numpy())

        return ab+c#abc


class up(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(up, self).__init__()
        # conv+bn
        self.up = QuanConv(in_c, out_c, kernel_size=1,norm=True, quan_input = False)
        
        # previous version
        #self.resize = QuanResize()
        self.resize = F.interpolate
        
        self.act = build_act("quanhswish")

    def forward(self, x, scale_x=None, scale_resize=None):
        x = self.up(x, scale_x)
        x = self.act(x)

        x_r_in = SymmetricQuantFunction.apply(x, 8, scale_resize) * scale_resize
        x_r_out = self.resize(x_r_in, scale_factor=2, mode='bilinear')
        x_r_q_out = SymmetricQuantFunction.apply(x_r_out, 8, scale_resize) * scale_resize
        
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()
            np.savez("npz_logging/" + str(idx) + "_resize", input_scale = scale_resize.detach().cpu().numpy(),
                                                            x = x.detach().cpu().numpy(),
                                                            x_r_in = x_r_in.detach().cpu().numpy(),
                                                            x_r_out = x_r_out.detach().cpu().numpy(),
                                                            x_r_q_out = x_r_q_out.detach().cpu().numpy())

        return x_r_q_out


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        # conv+bn
        self.down = QuanConv(in_c, out_c, kernel_size=2,
                             padding=0, stride=2, bias=False, norm=True, quan_input = False)
        self.act = build_act("quanhswish")

    def forward(self, x, scale_x=None):
        x = self.down(x, scale_x)
        x = self.act(x)
        return x


class block(nn.Module):
    def __init__(self, in_c, out_c,  dp=0, is_up=False, is_down=False, fuse=False, quan_input = True):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c, quan_input)
        else:
            self.fuse = QuanConv(in_c, out_c, kernel_size=1, stride=1)

        self.conv_out_quan_8_bit = LsqQuantizer4input(bit=8, per_channel=False, all_positive=False)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = conv(out_c, out_c, dropout_rate=dp) if self.in_c != self.out_c else conv(out_c, out_c, dropout_rate=dp, quan_input = quan_input)
        if self.is_up == True:
            self.up = up(out_c, out_c//2)
        if self.is_down == True:
            self.down = down(out_c, out_c*2)

    def forward(self, x, scale_x=None, scale_up=None):
        if self.in_c != self.out_c:
            x = self.fuse(x, scale_x)
            conv_out = self.conv(x)
        else:
            conv_out = self.conv(x, scale_x)
        
        q_conv_out, scale_conv_out = self.conv_out_quan_8_bit(conv_out)
        if self.is_up == False and self.is_down == False:
            return scale_conv_out, conv_out
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(q_conv_out, scale_conv_out, scale_up)
            return scale_conv_out, conv_out, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(q_conv_out, scale_conv_out)
            return scale_conv_out, conv_out, x_down
        else:
            x_up = self.up(q_conv_out, scale_conv_out, scale_up)
            x_down = self.down(q_conv_out, scale_conv_out)
            return scale_conv_out, conv_out, x_up, x_down


class FR_UNet_Quan(nn.Module):
    def __init__(self,  num_classes=1, num_channels=1, feature_scale=2,  dropout=0.2, fuse=True, out_ave=True, upsample="BiLinear"):
        super(FR_UNet_Quan, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse, quan_input = False) #Align conv_out scale
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse, quan_input = False)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse, quan_input = False)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse, quan_input = False)
        self.block12 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse, quan_input = False)
        self.block13 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse, quan_input = False)
        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse, quan_input = False)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse, quan_input = False)
        self.block21 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse, quan_input = False)
        self.block22 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse, quan_input = False)
        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse, quan_input = False)
        self.block31 = block(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse, quan_input = False)
        self.block40 = block(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.final1 = QuanConv(
            filters[0], 16, kernel_size=1, padding=0, bias=True, quan_input = False)
        self.final2 = QuanConv(
            filters[0], 16, kernel_size=1, padding=0, bias=True, quan_input = False)
        self.final3 = QuanConv(
            filters[0], 16, kernel_size=1, padding=0, bias=True, quan_input = False)
        self.final4 = QuanConv(
            filters[0], 16, kernel_size=1, padding=0, bias=True, quan_input = False)
        self.final5 = QuanConv(
            filters[0], 16, kernel_size=1, padding=0, bias=True)
        self.fuse = QuanConv(
            16, num_classes, kernel_size=1, padding=0, bias=True)
        
        # ? bit
        self.quan_ = nn.ModuleDict({
            name: LsqQuantizer4input(
                bit=16 if "16bit" in name else 8,
                per_channel=False,
                all_positive=False
            )
            for name in ["final1_16bit", "final2_16bit", "final3_16bit", "final4_16bit",
                        "final1_8bit", "final2_8bit", "final3_8bit", "final4_8bit", "final5_8bit"]
        })
        
        """
        self.quan_concat = nn.ModuleDict({
                name: LsqQuantizer4input(
                    bit=8,
                    per_channel=False,
                    all_positive=False 
                ) 
                for name in ["cat1", "cat2", "cat3", "cat4", "cat5",
                             "cat6", "cat7", "cat8", "cat9", "cat10",
                             "cat11"]
            })"
        """
        
        self.apply(InitWeights_He)

    def forward(self, x):
        s1_3, x1_3, x_down1_3 = self.block1_3(x)
        s1_2, x1_2, x_down1_2 = self.block1_2(SymmetricQuantFunction.apply(x1_3, 8, s1_3) * s1_3, s1_3)
        s2_2, x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3, scale_up = s1_2)
        
        #qx1_2, r1_2 = self.quan_concat["cat1"](x1_2)
        #qx_up2_2, r_up2_2 = self.quan_concat["cat1"](x_up2_2)
        #qcat1, os_cat1 = self.quan_concat["cat1"](torch.cat([x1_2, x_up2_2], dim=1))
        qcat1 = SymmetricQuantFunction.apply(torch.cat([x1_2, x_up2_2], dim=1), 8, s1_2) * s1_2
        os_cat1 = s1_2
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat1.detach().cpu().numpy())
        s1_1, x1_1, x_down1_1 = self.block1_1(qcat1, os_cat1)
        
        #qx_down1_2, rx_down1_2 = self.quan_concat["cat2"](x_down1_2)
        #qx2_2, rx2_2 = self.quan_concat["cat2"](x2_2)
        #qcat2, os_cat2 = self.quan_concat["cat2"](torch.cat([x_down1_2, x2_2], dim=1))
        qcat2 =  SymmetricQuantFunction.apply(torch.cat([x_down1_2, x2_2], dim=1), 8, s2_2) * s2_2
        os_cat2 = s2_2
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()    
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat2.detach().cpu().numpy())
        s2_1, x2_1, x_up2_1, x_down2_1 = self.block2_1(qcat2, os_cat2, scale_up = s1_1)
        
        s3_1, x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2, scale_up = s2_1)
        
        #qx1_1, rx1_1 = self.quan_concat["cat3"](x1_1)
        #qx_up2_1, rx_up2_1  = self.quan_concat["cat3"](x_up2_1)
        #qcat3, os_cat3 = self.quan_concat["cat3"](torch.cat([x1_1, x_up2_1], dim=1))
        qcat3 = SymmetricQuantFunction.apply(torch.cat([x1_1, x_up2_1], dim=1), 8, s1_1) * s1_1
        os_cat3 = s1_1
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()   
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat3.detach().cpu().numpy())
        s10, x10, x_down10 = self.block10(qcat3, os_cat3)
        
        #qx_down1_1, rx_down1_1 = self.quan_concat["cat4"](x_down1_1)
        #qx2_1, rx2_1 = self.quan_concat["cat4"](x2_1)
        #qx_up3_1, rx_up3_1 = self.quan_concat["cat4"](x_up3_1)
        #qcat4, os_cat4 = self.quan_concat["cat4"](torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        qcat4 = SymmetricQuantFunction.apply(torch.cat([x_down1_1, x2_1, x_up3_1], dim=1), 8, s2_1) * s2_1
        os_cat4 = s2_1
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()    
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat4.detach().cpu().numpy())
        s20, x20, x_up20, x_down20 = self.block20(qcat4, os_cat4, scale_up = s10)
        
        #qx_down2_1, rx_down2_1 = self.quan_concat["cat5"](x_down2_1)
        #qx3_1, rx3_1 = self.quan_concat["cat5"](x3_1)
        #qcat5, os_cat5 = self.quan_concat["cat5"](torch.cat([x_down2_1, x3_1], dim=1))
        qcat5 = SymmetricQuantFunction.apply(torch.cat([x_down2_1, x3_1], dim=1), 8, s3_1) * s3_1
        os_cat5 = s3_1
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()    
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat5.detach().cpu().numpy())
        s30, x30, x_up30 = self.block30(qcat5, os_cat5, scale_up = s20)
        
        s40, _, x_up40 = self.block40(x_down3_1, scale_up = s30)
        
        # qx10, rx10 = self.quan_concat["cat6"](x10)
        # qx_up20, rx_up20 = self.quan_concat["cat6"](x_up20)
        #qcat6, os_cat6 = self.quan_concat["cat6"](torch.cat([x10, x_up20], dim=1))
        qcat6 = SymmetricQuantFunction.apply(torch.cat([x10, x_up20], dim=1), 8, s10) * s10
        os_cat6 = s10
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()    
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat6.detach().cpu().numpy())
        s11, x11, x_down11 = self.block11(qcat6, os_cat6)
        
        # qx_down10, rx_down10 = self.quan_concat["cat7"](x_down10)
        # qx20, rx20 = self.quan_concat["cat7"](x20)
        # qx_up30, rx_up30 = self.quan_concat["cat7"](x_up30)
        #qcat7, os_cat7 = self.quan_concat["cat7"](torch.cat([x_down10, x20, x_up30], dim=1))
        qcat7 = SymmetricQuantFunction.apply(torch.cat([x_down10, x20, x_up30], dim=1), 8, s20) * s20
        os_cat7 = s20
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()    
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat7.detach().cpu().numpy())
        s21, x21, x_up21 = self.block21(qcat7, os_cat7, scale_up = s11)
        
        # qx_down20, rx_down20 = self.quan_concat["cat8"](x_down20)
        # qx30, rx30 = self.quan_concat["cat8"](x30)
        # qx_up40, rx_up40 = self.quan_concat["cat8"](x_up40)
        #qcat8, os_cat8 = self.quan_concat["cat8"](torch.cat([x_down20, x30, x_up40], dim=1))
        qcat8 = SymmetricQuantFunction.apply(torch.cat([x_down20, x30, x_up40], dim=1), 8, s30) * s30
        os_cat8 = s30
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()    
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat8.detach().cpu().numpy())
        x31, _, x_up31 = self.block31(qcat8, os_cat8, scale_up = s21)
        
        # qx11, rx11 = self.quan_concat["cat9"](x11)
        # qx_up21, rx_up21 = self.quan_concat["cat9"](x_up21)
        #qcat9, os_cat9 = self.quan_concat["cat9"](torch.cat([x11, x_up21], dim=1))
        qcat9 = SymmetricQuantFunction.apply(torch.cat([x11, x_up21], dim=1), 8, s11) * s11
        os_cat9 = s11
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()    
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat9.detach().cpu().numpy())
        s12, x12 = self.block12(qcat9, os_cat9)
        
        # qx_down11, rx_down11 = self.quan_concat["cat10"](x_down11) 
        # qx21, rx21 = self.quan_concat["cat10"](x21)
        # qx_up31, rx_up31 = self.quan_concat["cat10"](x_up31)
        #qcat10, os_cat10 = self.quan_concat["cat10"](torch.cat([x_down11, x21, x_up31], dim=1)) 
        qcat10 = SymmetricQuantFunction.apply(torch.cat([x_down11, x21, x_up31], dim=1), 8, s21) * s21
        os_cat10 = s21
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()    
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat10.detach().cpu().numpy())
        s22, _, x_up22 = self.block22(qcat10, os_cat10, scale_up = s12)
        
        # qx12, rx12 = self.quan_concat["cat11"](x12)
        # qx_up22, rx_up22 = self.quan_concat["cat11"](x_up22)
        #qcat11, os_cat11 = self.quan_concat["cat11"](torch.cat([x12, x_up22], dim=1))
        qcat11 = SymmetricQuantFunction.apply(torch.cat([x12, x_up22], dim=1), 8, s12) * s12
        os_cat11 = s12
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()    
            np.savez("npz_logging/" + str(idx)+ "_concat", output_scale=os_cat11.detach().cpu().numpy())
        s13, x13 = self.block13(qcat11, os_cat11)    
        
        if self.out_ave == True:
            q_x1_1 = SymmetricQuantFunction.apply(x1_1, 8, os_cat3) * os_cat3
            a, r_a = self.quan_["final1_16bit"](self.final1(q_x1_1, os_cat3)) # 8, 16
            q_x10 = SymmetricQuantFunction.apply(x10, 8, os_cat6) * os_cat6
            b, r_b = self.quan_["final1_8bit"](self.final2(q_x10, os_cat6))        
            
            #b, _ = self.quan_["final1_16bit"](b) # No need to unifi add node output scale
            ab, r_ab = self.quan_["final2_8bit"](a+b)    # 8, 16
            if not self.training and get_global_idx() >= 0: #log npz:
                idx = get_next_global_idx()
                np.savez("npz_logging/" + str(idx) + "_add", a_scale=r_a.detach().cpu().numpy(),
                                                             b_scale=r_b.detach().cpu().numpy(),
                                                             a_bitdepth = 16,
                                                             b_bitdepth = 8)
            
            q_x11 = SymmetricQuantFunction.apply(x11, 8, os_cat9) * os_cat9
            c, r_c = self.quan_["final2_16bit"](self.final3(q_x11, os_cat9))
            #ab, _ = self.quan_["final2_16bit"](ab) # Add Node output can only be 8-bit and already quantized in line 372
            abc, r_abc = self.quan_["final3_8bit"](ab+c) # 8, 16
            if not self.training and get_global_idx() >= 0: #log npz:
                idx = get_next_global_idx()
                np.savez("npz_logging/" + str(idx) + "_add", a_scale=r_ab.detach().cpu().numpy(),
                                                             b_scale=r_c.detach().cpu().numpy(),
                                                             a_bitdepth = 8,
                                                             b_bitdepth = 16)  
            
            q_x12 = SymmetricQuantFunction.apply(x12, 8, os_cat11) * os_cat11
            d, r_d = self.quan_["final3_16bit"](self.final4(q_x12, os_cat11)) 
            #abc, _ = self.quan_["final3_16bit"](abc) # Add Node output can only be 8-bit and already quantized in line 382
            abcd, r_abcd = self.quan_["final4_8bit"](abc+d) # 8, 16
            if not self.training and get_global_idx() >= 0: #log npz:
                idx = get_next_global_idx()
                np.savez("npz_logging/" + str(idx) + "_add", a_scale=r_abc.detach().cpu().numpy(),
                                                             b_scale=r_d.detach().cpu().numpy(),
                                                             a_bitdepth = 8,
                                                             b_bitdepth = 16) 
                          
            e, r_e = self.quan_["final4_16bit"](self.final5(x13))
            #abcd, _ = self.quan_["final4_16bit"](abcd) # Add Node output can only be 8-bit and already quantized in line 392
            #output, r_out = self.quan_["final5_8bit"]((abcd+e)/5)
            #output, r_out = self.quan_["final5_8bit"]((abcd+e)) # leave quan for final conv
            if not self.training and get_global_idx() >= 0: #log npz:
                idx = get_next_global_idx()
                np.savez("npz_logging/" + str(idx) + "_add", a_scale=r_abcd.detach().cpu().numpy(),
                                                             b_scale=r_e.detach().cpu().numpy(),
                                                             a_bitdepth = 8,
                                                             b_bitdepth = 16) 
                
            output, r_out = self.quan_["final5_8bit"](self.fuse(abcd+e))

            if not self.training and get_global_idx() >= 0: #log npz:
                np.savez("npz_logging/output", out_scale=r_out.detach().cpu().numpy()) 
            
            '''
            if not self.training and get_global_idx() >= 0: #log npz:
                idx = get_next_global_idx()
                np.savez("npz_logging/" + str(idx) + "_div", out_scale=r_out.detach().cpu().numpy())'
            '''
            
            # output = self.final1(x1_1)+self.final2(x10)+self.final3(x11)+self.final4(x12)+self.final5(x13)/5
        else:
            output = self.final5(x13)
            

        return output
