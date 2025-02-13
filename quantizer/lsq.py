from math import e
from matplotlib import scale
from matplotlib.pyplot import sca
import torch

import numpy as np
import decimal
from fractions import Fraction
from decimal import Decimal

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def trunc_pass(x):
    y = torch.trunc(x)
    y_grad = x
    return (y - y_grad).detach() + y_grad

def clip(x, eps):
    x_clip = torch.where(x > eps.to(x.device), x, eps.to(x.device))
    return x - x.detach() + x_clip.detach()

def batch_frexp(inputs, bit=8):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Parameters:
    ----------
    inputs: scaling factor
    return: (mantissa, exponent)
    """
    shape_of_input = inputs.size()

    # transform the input to be a 1-d tensor
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().detach().numpy())

    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(m * (2 ** bit)).quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = 1.0*bit - output_e

    return torch.from_numpy(output_m).to(inputs.device).view(shape_of_input), \
           torch.from_numpy(output_e).to(inputs.device).view(shape_of_input)
           
class LsqQuantizer4input(torch.nn.Module):
    def __init__(self, bit=8, update=False, head=None, all_positive=False, per_channel=True, learnable = True, **kwargs):
        super(LsqQuantizer4input, self).__init__()

        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        self.update = update
        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        self.head = head
        if self.head is not None:
            self.s = torch.nn.Parameter(torch.ones((head)), requires_grad=True)
        else:
            self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.register_buffer('initialized_alpha', torch.zeros(1))

    def init_from(self, x, modified_init=True, pr=False, *args, **kwargs):
        if not modified_init: 
            print("lsq init")
            init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
        else:
            if self.head is not None: # B, head, N, C
                init_val_max = x.detach().abs().amax(dim=(0,2,3)) / self.thd_pos
                init_val_lsq = x.detach().abs().mean(dim=(0,2,3)) * 2 / (self.thd_pos ** 0.5)
            else:
                init_val_max = x.detach().abs().max() / self.thd_pos
                init_val_lsq = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            if self.bit >= 8:
                init_val = torch.where(init_val_max < init_val_lsq, init_val_max, init_val_lsq)
            else:
                init_val = init_val_lsq
        self.s.data.copy_(init_val)
        self.initialized_alpha.fill_(1) # only initialize once, first forward pass
        
    def forward(self, x, modified_init=True, pr=False):
        # return x, None
        if self.update == False:
            if self.initialized_alpha == 0:
                print("lsq init_input begin")
                self.init_from(x, modified_init=modified_init, pr=pr)     
        else:
            self.init_from(x, modified_init=modified_init, pr=pr)
        alpha = self.s # TODO   
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(clip(alpha.to(x.device), torch.tensor(1e-5, device=x.device).float()), s_grad_scale) # TODO, clip
        
        # with dyadic
        pow = torch.round(torch.log2(s_scale.detach()))
        clip_val = torch.pow(2, pow)
        scale = (clip_val - s_scale).detach() + s_scale
        # without dyadic
        # scale = s_scale
        
        if self.head is not None:
            scale_view = scale.view(1,-1,1,1).expand_as(x) # B, h, N, C
        else:
            scale_view = scale
        x = x / scale_view
        
        if self.bit == 1 and not self.all_positive:
            x = torch.sign(x)
        else:
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
        x = x * scale_view
        return x, scale


class LsqQuantizer4weight(torch.nn.Module):
    def __init__(self, bit, all_positive=False, per_channel=True, learnable = True, per_channel_num=1, **kwargs):
        super(LsqQuantizer4weight, self).__init__()

        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        # you need to register the parameter names earlier
        if self.per_channel:
            if per_channel_num is None:
                raise NotImplementedError("per_channel_num must be specified when per_channel is True")
            self.s = torch.nn.Parameter(torch.ones(per_channel_num), requires_grad=True)
        else:
            self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.register_buffer('initialized_alpha', torch.zeros(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            if len(x.shape) == 1: # bias
                init_val = x.detach().abs() / self.thd_pos
            elif len(x.shape) == 2: # linear weight [out, in]
                init_val = 2 * x.detach().abs().mean(dim=-1) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=-1) / (self.thd_pos ** 0.5)
            elif len(x.shape) == 4: # conv weight [out, in, k, k]
                init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=-1) / (self.thd_pos ** 0.5) if not self.all_positive \
                        else 4 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=-1) / (self.thd_pos ** 0.5)
        else: # per_layer quantization
            init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
        self.s.data.copy_(init_val)
        self.initialized_alpha.fill_(1)   # only initialize once, first forward pass
        
    def forward(self, x, pr=False):
        # return x, None
        # step size
        if self.per_channel:
            if self.initialized_alpha == 0:
                # print("lsq init_weight begin")
                self.init_from(x)
            if len(x.shape) != 1: alpha = torch.unsqueeze(self.s,dim=-1) # shape [out, 1]
            else: alpha = self.s
        else:
            if self.initialized_alpha == 0:
                self.init_from(x)     
            alpha = self.s       
        # step size gradient scale
        if self.per_channel:
            if len(x.shape) == 2:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[-1]) ** 0.5)
            elif len(x.shape) == 4:
                s_grad_scale = 1.0 / ((self.thd_pos * x.shape[-3]*x.shape[-2]* x.shape[-1]) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        # scale
        # s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale) # TODO
        s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale) # TODO
        
        # dyadic
        scale_m, scale_e =  batch_frexp(s_scale.detach(), bit=8)
        scale = (scale_m / torch.pow(2, scale_e)).type(torch.float32)
        scale_new = (scale - s_scale).detach() + s_scale
        # without dyadic
        # if len(x.shape) == 4:
        #     scale_new = s_scale.view(s_scale.shape[0],1,1,1)
        # else: scale_new = s_scale
        if len(x.shape) == 2:
            scale_new = scale_new.view(scale_new.shape[0],1)
        elif len(x.shape) == 4:
            scale_new = scale_new.view(scale_new.shape[0],1,1,1)
            
        # quantize
        x = x / scale_new.to(x.device)
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * scale_new
        return x, scale_new

class VSQQuantizer4input(torch.nn.Module):
    def __init__(self, N, C, tile_n=16, tile_c=8, head=None, vsq_mode='normal', bit=8, all_positive=False, per_channel=True, learnable = True):
        super(VSQQuantizer4input, self).__init__()
        self.vsq_mode = vsq_mode
        self.N, self.C, self.tile_n, self.tile_c, self.head = N, C, tile_n, tile_c, head
        if self.N % self.tile_n != 0: self.tile_n = self.N
        if self.C % self.tile_c != 0: self.tile_c = self.C
        # self.tile_n, self.tile_c = N, C
        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        self.num_tiles_n = self.N // self.tile_n
        self.num_tiles_c = self.C // self.tile_c
        if self.head is None:
            self.s = torch.nn.Parameter(torch.ones(self.num_tiles_n, self.num_tiles_c), requires_grad=True)
        else:
            self.s = torch.nn.Parameter(torch.ones(self.head, self.num_tiles_n, self.num_tiles_c), requires_grad=True)
        self.register_buffer('initialized_alpha', torch.zeros(1))
        self.loss = torch.nn.MSELoss()

    def init_from(self, x, modified_init=True):
        if not modified_init: 
            if self.head is None:
                init_val = x.detach().abs().mean(dim=(0,3,4)) * 2 / (self.thd_pos ** 0.5)
            else:
                init_val = x.detach().abs().mean(dim=(0,4,5)) * 2 / (self.thd_pos ** 0.5)
        else:
            if self.head is None:
                init_val_max = x.detach().abs().amax(dim=(0,3,4)) / self.thd_pos
                init_val_lsq = x.detach().abs().mean(dim=(0,3,4)) * 2 / (self.thd_pos ** 0.5)
            else:
                init_val_max = x.detach().abs().amax(dim=(0,4,5)) / self.thd_pos
                init_val_lsq = x.detach().abs().mean(dim=(0,4,5)) * 2 / (self.thd_pos ** 0.5)
            init_val = torch.where(init_val_max < init_val_lsq, init_val_max, init_val_lsq)
            init_val = init_val_lsq
        self.s.data.copy_(init_val)
        self.initialized_alpha.fill_(1) # only initialize once, first forward pass
    
    # def reshape(self, x):
    #     if self.head is None:
    #         B, N, C = x.shape
    #         assert N == self.N and C == self.C
    #         tile_n, tile_c = self.tile_n, self.tile_c
    #         num_tiles_n, num_tiles_c = self.num_tiles_n, self.num_tiles_c
    #         x = x.reshape(B, num_tiles_n, tile_n, C) # B, num_tiles_n, tile_n, C
    #         x = x.reshape(B, num_tiles_n, tile_n, num_tiles_c, tile_c) # B, num_tiles_n, tile_n, num_tiles_c, tile_c
    #         x = x.permute(0, 1, 3, 2, 4) # B, num_tiles_n, num_tiles_c, tile_n, tile_c
    #     else:
    #         B, head, N, C = x.shape
    #         assert N == self.N and C == self.C and head == self.head
    #         tile_n, tile_c = self.tile_n, self.tile_c
    #         num_tiles_n, num_tiles_c = self.num_tiles_n, self.num_tiles_c
    #         head = self.head
    #         x = x.reshape(B, head, num_tiles_n, tile_n, C) # B, head, num_tiles_n, tile_n, C
    #         x = x.reshape(B, head, num_tiles_n, tile_n, num_tiles_c, tile_c) # B, head, num_tiles_n, tile_n, num_tiles_c, tile_c
    #         x = x.permute(0, 1, 2, 4, 3, 5) # B, head, num_tiles_n, num_tiles_c, tile_n, tile_c
    #     return x
    
    # def re_reshape(self, x):
    #     if self.head is None:
    #         B, num_tiles_n, num_tiles_c, tile_n, tile_c = x.shape
    #         x = x.permute(0, 1, 3, 2, 4) # B, num_tiles_n, tile_n, num_tiles_c, tile_c
    #         x = x.reshape(B, num_tiles_n, tile_n, num_tiles_c*tile_c) # B, num_tiles_n, tile_n, C
    #         x = x.reshape(B, num_tiles_n*tile_n, num_tiles_c*tile_c) # B, N, C
    #     else:
    #         B, head, num_tiles_n, num_tiles_c, tile_n, tile_c = x.shape
    #         x = x.permute(0, 1, 2, 4, 3, 5) # B, head, num_tiles_n, tile_n, num_tiles_c, tile_c
    #         x = x.reshape(B, head, num_tiles_n, tile_n, num_tiles_c*tile_c) # B, head, num_tiles_n, tile_n, C
    #         x = x.reshape(B, head, num_tiles_n*tile_n, num_tiles_c*tile_c) # B, head, N, C
    #     return x
    
    def reshape(self, x):
        if self.head is None:
            B, N, C = x.shape
            assert N == self.N and C == self.C
            tile_n, tile_c = self.tile_n, self.tile_c
            num_tiles_n, num_tiles_c = self.num_tiles_n, self.num_tiles_c
            
            x = x.reshape(B, num_tiles_n, tile_n, num_tiles_c, tile_c)  # B, num_tiles_n, tile_n, num_tiles_c, tile_c
            x = x.permute(0, 1, 3, 2, 4)  # B, num_tiles_n, num_tiles_c, tile_n, tile_c
        else:
            B, head, N, C = x.shape
            assert N == self.N and C == self.C and head == self.head
            tile_n, tile_c = self.tile_n, self.tile_c
            num_tiles_n, num_tiles_c = self.num_tiles_n, self.num_tiles_c

            x = x.reshape(B, head, num_tiles_n, tile_n, num_tiles_c, tile_c)  # B, head, num_tiles_n, tile_n, num_tiles_c, tile_c
            x = x.permute(0, 1, 2, 4, 3, 5)  # B, head, num_tiles_n, num_tiles_c, tile_n, tile_c
        return x

    def re_reshape(self, x):
        if self.head is None:
            B, num_tiles_n, num_tiles_c, tile_n, tile_c = x.shape
            x = x.permute(0, 1, 3, 2, 4)  # B, num_tiles_n, tile_n, num_tiles_c, tile_c
            x = x.reshape(B, num_tiles_n * tile_n, num_tiles_c * tile_c)  # B, N, C
        else:
            B, head, num_tiles_n, num_tiles_c, tile_n, tile_c = x.shape
            x = x.permute(0, 1, 2, 4, 3, 5)  # B, head, num_tiles_n, tile_n, num_tiles_c, tile_c
            x = x.reshape(B, head, num_tiles_n * tile_n, num_tiles_c * tile_c)  # B, head, N, C
        return x

        
    def forward(self, x, modified_init=True):
        if len(x.shape) != 3 and (len(x.shape) != 4 or (self.head is None and self.vsq_mode!='bchw')): 
            raise NotImplementedError("Currently input shape must be 3D tensor")
        if self.vsq_mode == 'bchw': 
            B, C, H, W = x.shape
            x = x.reshape(B, C, H*W).permute(0, 2, 1)
        x = self.reshape(x)
        # return x, None
        if self.initialized_alpha == 0:
            print("vsq init_input begin")
            self.init_from(x, modified_init=modified_init)     
        alpha = self.s.abs() # TODO   
        s_grad_scale = 1.0 / ((self.thd_pos * self.tile_n * self.tile_c) ** 0.5) # TODO per vector
        s_scale = grad_scale(clip(alpha.to(x.device), torch.tensor(1e-10, device=x.device).float()), s_grad_scale) # TODO, clip
        
        # with dyadic
        pow = torch.round(torch.log2(s_scale.detach()))
        clip_val = torch.pow(2, pow)
        scale = (clip_val - s_scale).detach() + s_scale
        # without dyadic
        # scale = s_scale
        
        scale_expand = scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x = x / scale_expand
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * scale_expand
        x = self.re_reshape(x)
        if self.vsq_mode == 'bchw':
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x, scale, None

class VSQQuantizer4weight(torch.nn.Module):
    def __init__(self, N, C, K=1, tile_n=8, tile_c=8, bit=8, vsq_mode='normal', all_positive=False): # N是out channels C是in channels
        super(VSQQuantizer4weight, self).__init__()
        self.vsq_mode = vsq_mode
        self.N, self.C, self.K, self.tile_n, self.tile_c = N, C, K, tile_n, tile_c
        if self.N % self.tile_n != 0: self.tile_n = self.N
        if self.C % self.tile_c != 0: self.tile_c = self.C
        # self.tile_n, self.tile_c = 1, C
        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        self.bit = bit
        self.all_positive = all_positive
        self.num_tiles_n = self.N // self.tile_n
        self.num_tiles_c = self.C // self.tile_c
        self.s = torch.nn.Parameter(torch.ones(self.num_tiles_n, self.num_tiles_c), requires_grad=True)
        self.register_buffer('initialized_alpha', torch.zeros(1))

    def init_from(self, x, modified_init=True):
        if not modified_init: 
            init_val = x.detach().abs().mean(dim=tuple(range(2, x.dim()))) * 2 / (self.thd_pos ** 0.5)
        else:
            init_val_max = x.detach().abs().amax(dim=tuple(range(2, x.dim()))) / self.thd_pos
            init_val_lsq = x.detach().abs().mean(dim=tuple(range(2, x.dim()))) * 2 / (self.thd_pos ** 0.5)
            init_val = torch.where(init_val_max < init_val_lsq, init_val_max, init_val_lsq)
            init_val = init_val_lsq # important!
        self.s.data.copy_(init_val)
        self.initialized_alpha.fill_(1) # only initialize once, first forward pass
        
    # def reshape(self, x):
    #     if self.vsq_mode != '3d':
    #         N, C = x.shape
    #         assert N == self.N and C == self.C
    #         tile_n, tile_c = self.tile_n, self.tile_c
    #         num_tiles_n, num_tiles_c = self.num_tiles_n, self.num_tiles_c
    #         x = x.reshape(num_tiles_n, tile_n, C) # num_tiles_n, tile_n, C
    #         x = x.reshape(num_tiles_n, tile_n, num_tiles_c, tile_c) # num_tiles_n, tile_n, num_tiles_c, tile_c
    #         x = x.permute(0, 2, 1, 3) # num_tiles_n, num_tiles_c, tile_n, tile_c
    #     else:
    #         N, C, K, _ = x.shape
    #         assert N == self.N and C == self.C and K == self.K
    #         tile_n, tile_c = self.tile_n, self.tile_c
    #         num_tiles_n, num_tiles_c = self.num_tiles_n, self.num_tiles_c
    #         x = x.reshape(num_tiles_n, tile_n, C, K, K) # num_tiles_n, tile_n, C, K, K
    #         x = x.reshape(num_tiles_n, tile_n, num_tiles_c, tile_c, K, K) # num_tiles_n, tile_n, num_tiles_c, tile_c, K, K
    #         x = x.permute(0, 2, 1, 3, 4, 5) # num_tiles_n, num_tiles_c, tile_n, tile_c, K, K
    #     return x
    
    # def re_reshape(self, x):
    #     if self.vsq_mode != '3d':
    #         num_tiles_n, num_tiles_c, tile_n, tile_c = x.shape
    #         x = x.permute(0, 2, 1, 3) # num_tiles_n, tile_n, num_tiles_c, tile_c
    #         tile_n, tile_c = self.tile_n, self.tile_c
    #         num_tiles_n, num_tiles_c = self.num_tiles_n, self.num_tiles_c
    #         x = x.reshape(num_tiles_n, tile_n, num_tiles_c*tile_c) # num_tiles_n, tile_n, C
    #         x = x.reshape(num_tiles_n*tile_n, num_tiles_c*tile_c) # N, C
    #     else:
    #         num_tiles_n, num_tiles_c, tile_n, tile_c, K, _ = x.shape
    #         x = x.permute(0, 2, 1, 3, 4, 5) # num_tiles_n, tile_n, num_tiles_c, tile_c, K, K
    #         tile_n, tile_c = self.tile_n, self.tile_c
    #         num_tiles_n, num_tiles_c = self.num_tiles_n, self.num_tiles_c
    #         x = x.reshape(num_tiles_n, tile_n, num_tiles_c*tile_c, K, K) # num_tiles_n, tile_n, C, K, K
    #         x = x.reshape(num_tiles_n*tile_n, num_tiles_c*tile_c, K, K) # N, C, K, K
    #     return x

    def reshape(self, x):
        if self.vsq_mode != '3d':
            N, C = x.shape
            assert N == self.N and C == self.C
            tile_n, tile_c = self.tile_n, self.tile_c
            num_tiles_n, num_tiles_c = self.num_tiles_n, self.num_tiles_c
            
            x = x.reshape(num_tiles_n, tile_n, num_tiles_c, tile_c)  # num_tiles_n, tile_n, num_tiles_c, tile_c
            x = x.permute(0, 2, 1, 3)  # num_tiles_n, num_tiles_c, tile_n, tile_c
        else:
            N, C, K, _ = x.shape
            assert N == self.N and C == self.C and K == self.K
            tile_n, tile_c = self.tile_n, self.tile_c
            num_tiles_n, num_tiles_c = self.num_tiles_n, self.num_tiles_c
            
            x = x.reshape(num_tiles_n, tile_n, num_tiles_c, tile_c, K, K)  # num_tiles_n, tile_n, num_tiles_c, tile_c, K, K
            x = x.permute(0, 2, 1, 3, 4, 5)  # num_tiles_n, num_tiles_c, tile_n, tile_c, K, K
        return x

    def re_reshape(self, x):
        if self.vsq_mode != '3d':
            num_tiles_n, num_tiles_c, tile_n, tile_c = x.shape
            x = x.permute(0, 2, 1, 3)  # num_tiles_n, tile_n, num_tiles_c, tile_c
            
            x = x.reshape(num_tiles_n * tile_n, num_tiles_c * tile_c)  # N, C
        else:
            num_tiles_n, num_tiles_c, tile_n, tile_c, K, _ = x.shape
            x = x.permute(0, 2, 1, 3, 4, 5)  # num_tiles_n, tile_n, num_tiles_c, tile_c, K, K
            
            x = x.reshape(num_tiles_n * tile_n, num_tiles_c * tile_c, K, K)  # N, C, K, K
        return x


    def forward(self, x):
        if self.vsq_mode == 'pw': x = x.reshape(self.N, self.C)
        x = self.reshape(x)
        if self.initialized_alpha == 0:
            print("vsq init_weight begin")
            self.init_from(x)
        alpha = self.s.abs() # TODO
        # step size gradient scale
        # s_grad_scale = 1.0 / ((self.thd_pos * x.shape[-1]) ** 0.5) # TODO
        s_grad_scale = 1.0 / ((self.thd_pos * self.tile_n * self.tile_c * self.K * self.K) ** 0.5)
        # scale
        s_scale = grad_scale(clip(alpha, torch.tensor(1e-5).float().to(x.device)), s_grad_scale)
        
        # dyadic
        scale_m, scale_e =  batch_frexp(s_scale.detach(), bit=8)
        scale = (scale_m / torch.pow(2, scale_e)).type(torch.float32)
        scale_new = (scale - s_scale).detach() + s_scale
        # without dyadic
        # scale_new = s_scale
        
        # quantize
        if self.vsq_mode != '3d': scale_new_expand = scale_new.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        else: scale_new_expand = scale_new.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x = x / scale_new_expand
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * scale_new_expand
        x = self.re_reshape(x)
        if self.vsq_mode == 'pw': x = x.reshape(self.N, self.C, 1, 1)
        return x, scale_new
