import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .lsq import *
import warnings

global_idx = 0

def get_next_global_idx():
    global global_idx
    global_idx = global_idx + 1
    return global_idx

def reset_global_idx():
    global global_idx
    global_idx = 0
    
def get_global_idx():
    global global_idx
    return global_idx

## TOcheck
class QuanLeakyRELU(nn.LeakyReLU):

    def __init__(self, negative_slope, inplace=True, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1):
        super(QuanLeakyRELU, self).__init__()
        self.quan_input = quan_input
        self.mode = mode
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                self.lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = self.lsq_a
            elif self.mode == 'vsq':
                self.vsq_a  = VSQQuantizer4input(
                                N=N, C=C,
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = self.vsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        self.leakyrelu = nn.LeakyReLU(negative_slope, inplace)
        
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x, = self.quan_a(input)
        return self.leakyrelu(input)

class QuanGELU(nn.GELU):

    def __init__(self, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1):
        super(QuanGELU, self).__init__()
        self.quan_input = quan_input
        self.mode = mode
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                self.lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = self.lsq_a
            elif self.mode == 'vsq':
                self.vsq_a  = VSQQuantizer4input(
                                N=N, C=C,
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = self.vsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x, = self.quan_a(input)
        return F.gelu(input)

class QuanRELU(nn.GELU):

    def __init__(self, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1):
        super(QuanRELU, self).__init__()
        self.quan_input = quan_input
        self.mode = mode
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                self.lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = self.lsq_a
            elif self.mode == 'vsq':
                self.vsq_a  = VSQQuantizer4input(
                                N=N, C=C,
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = self.vsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x, = self.quan_a(input)
        return F.relu(input)
    
class QuanSoftmax(nn.Softmax):

    def _init__(self, dim, quan_input_sm=True, nbit_a=8, head=None):
        super(QuanSoftmax, self).__init__(dim)
        self.quan_input = quan_input_sm
        if self.quan_input:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                            bit=self.nbit_a, head=head,
                            all_positive=False,
                            per_channel=False)
            self.quan_a = self.lsq_a
        
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x = self.quan_a(input)
        return F.softmax(input, self.dim, _stacklevel=5)
    
class QuanLayerNorm(nn.LayerNorm): 

    def __init__(self, normalized_shape, quan_input=True, nbit_a=8):
        super(QuanLayerNorm, self).__init__(normalized_shape)
        self.quan_input = quan_input
        if self.quan_input:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                            bit=self.nbit_a,
                            all_positive=False,
                            per_channel=False)
            self.quan_a = self.lsq_a
        
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x = self.quan_a(input)
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

## TO CHECK   
class QuanBatchNorm(nn.BatchNorm2d): 
    def __init__(self, num_features, quan_input=True, nbit_a=8):
        super(QuanBatchNorm, self).__init__(num_features)
        self.quan_input = quan_input
        self.num_features = num_features
        if self.quan_input:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                            bit=self.nbit_a,
                            all_positive=False,
                            per_channel=False)
            self.quan_a = self.lsq_a
        
        self.batch2d = nn.BatchNorm2d(num_features, eps=self.eps)
        
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x = self.quan_a(input)
        return self.batch2d(input)
        
class QuanBertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1):
        super(QuanBertLayerNorm, self).__init__()
        self.quan_input = quan_input
        self.mode = mode
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                self.lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = self.lsq_a
            elif self.mode == 'vsq':
                self.vsq_a  = VSQQuantizer4input(
                                N=N, C=C, tile_c=C,
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = self.vsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, scale_x=None):
        if self.quan_input:
            x, scale_x, = self.quan_a(x)
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class QuanMMHead(nn.Module):

    def __init__(self, quan_input_a=True, quan_input_b=True, nbit_a=8, head=None):
        super(QuanMMHead, self).__init__()
        self.quan_input_a = quan_input_a
        self.quan_input_b = quan_input_b
        if self.quan_input_a:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                            bit=self.nbit_a, head=head,
                            all_positive=False,
                            per_channel=False)
            self.quan_a = self.lsq_a
            
        if self.quan_input_b:
            self.nbit_a = nbit_a
            self.lsq_b  = LsqQuantizer4input(
                            bit=self.nbit_a, head=head,
                            all_positive=False,
                            per_channel=False)
            self.quan_b = self.lsq_b
        
    def forward(self, input_a, input_b, scale_a=None, scale_b=None):
        if self.quan_input_a:
            input_a, scale_a = self.quan_a(input_a)
        if self.quan_input_b:
            input_b, scale_b = self.quan_b(input_b)
        return input_a @ input_b

class QuanEWM(nn.Module):
    """
    Quantized element-wise multiplication 
    """
    def __init__(self, quan_input_a=True, quan_input_b=True, nbit_a=8, head=None):
        super(QuanEWM, self).__init__()
        self.quan_input_a = quan_input_a
        self.quan_input_b = quan_input_b
        if self.quan_input_a:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                            bit=self.nbit_a, head=head,
                            all_positive=False,
                            per_channel=False)
            self.quan_a = self.lsq_a
            
        if self.quan_input_b:
            self.nbit_a = nbit_a
            self.lsq_b  = LsqQuantizer4input(
                            bit=self.nbit_a, head=head,
                            all_positive=False,
                            per_channel=False)
            self.quan_b = self.lsq_b
        
    def forward(self, input_a, input_b, scale_a=None, scale_b=None):
        if self.quan_input_a:
            input_a, scale_a = self.quan_a(input_a)
        if self.quan_input_b:
            input_b, scale_b = self.quan_b(input_b)
        return input_a * input_b
    
class QuanResidual(nn.Module):

    def __init__(self, quan_input_res=True, quan_input_out=True, nbit_a=8):
        super(QuanResidual, self).__init__()
        self.quan_input_res = quan_input_res
        self.quan_input_out = quan_input_out
        if self.quan_input_res:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                            bit=self.nbit_a,
                            all_positive=False,
                            per_channel=False)
            self.quan_res = self.lsq_a
        
    def forward(self, input_res, input_out, scale_a=None, scale_b=None):
        if self.quan_input_res:
            input_res, scale_res = self.quan_res(input_res)
        if self.quan_input_out:
            input_out, scale_out = self.quan_res(input_out) # TODO here
        return input_res + input_out

class QuanResize(nn.Module):

    def __init__(self, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1):
        super(QuanResize, self).__init__()
        self.quan_input = quan_input
        self.mode = mode
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                self.lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = self.lsq_a
            elif self.mode == 'vsq':
                self.vsq_a  = VSQQuantizer4input(
                                N=N, C=C, vsq_mode='bchw',
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = self.vsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
            
    def forward(self, input,
            size=None,
            scale_factor=None,
            mode='nearest',
            align_corners=None,
            warning=True):
        if warning:
            if size is not None and align_corners:
                input_h, input_w = tuple(int(x) for x in input.shape[2:])
                output_h, output_w = tuple(int(x) for x in size)
                if output_h > input_h or output_w > output_h:
                    if ((output_h > 1 and output_w > 1 and input_h > 1
                        and input_w > 1) and (output_h - 1) % (input_h - 1)
                            and (output_w - 1) % (input_w - 1)):
                        warnings.warn(
                            f'When align_corners={align_corners}, '
                            'the output would more aligned if '
                            f'input size {(input_h, input_w)} is `x+1` and '
                            f'out size {(output_h, output_w)} is `nx+1`')
        if isinstance(size, torch.Size):
            size = tuple(int(x) for x in size)
        if self.quan_input:
            input, scale_x = self.quan_a(input)
        return F.interpolate(input, size, scale_factor, mode, align_corners)

class QuanMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., batch_first=False, kdim=None, vdim=None, 
                 bias=True, add_bias_kv=False, nbit_w=8, nbit_a=8, mode='lsq', quan_input=True, finetune_test=False):
        super(QuanMultiheadAttention, self).__init__(
            embed_dim, num_heads, dropout, batch_first=batch_first, kdim=kdim, vdim=vdim, bias=bias, add_bias_kv=add_bias_kv)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.finetune_test = finetune_test
        # qkv in one nn.linear
        if self._qkv_same_embed_dim:
            self.in_w = torch.zeros(3 * embed_dim, embed_dim)
        else:
            self.in_w_q = torch.zeros(embed_dim, embed_dim)
            self.in_w_k = torch.zeros(embed_dim, self.kdim)
            self.in_w_v = torch.zeros(embed_dim, self.vdim)
        if bias:
            self.in_b = torch.zeros(3 * embed_dim)
        assert add_bias_kv == False, 'Not implemented add_bias_kv yet'
        # out linear
        self.out_b = torch.zeros(self.vdim)
        self.out_w = torch.zeros(self.vdim, self.vdim)
        # quantization definition
        self.quan_input = quan_input
        self.quan_q = self.quan_k = \
        self.quan_v = self.quan_o = \
        self.quan_out = LsqQuantizer4input(
                        bit=self.nbit_a, 
                        all_positive=False, 
                        per_channel=False)
        self.mode = mode 
        self.quan_w4q = self.quan_w4k = \
        self.quan_w4v = LsqQuantizer4weight(
                            bit=self.nbit_w, 
                            all_positive=False, 
                            per_channel=True, 
                            per_channel_num=self.embed_dim)
        self.quan_w4out = LsqQuantizer4weight(
                            bit=self.nbit_w, 
                            all_positive=False, 
                            per_channel=True, 
                            per_channel_num=self.out_proj.weight.shape[0])
        self.quan_bf_sm = LsqQuantizer4input(
                            bit=self.nbit_a, 
                            all_positive=False, 
                            per_channel=False)
        self.softmax = nn.Softmax(dim=-1)
        self.qk_mul = QuanMMHead(quan_input_a=True, quan_input_b=True, nbit_a=8)
        self.attnv_mul = QuanMMHead(quan_input_a=True, quan_input_b=True, nbit_a=8)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, query, key, value, scale_q=None, scale_k=None, scale_v=None, 
                key_padding_mask=None, attn_mask=None):
        r"""
    Args:
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        """
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)
        if self.batch_first:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        N, B_, C = query.shape
        src_len = key.size(0)
         # update source sequence length after adjustments
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B_, src_len), \
                f"expecting key_padding_mask shape of {(B_, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(B_, 1, 1, src_len).  \
                expand(-1, self.num_heads, -1, -1).reshape(B_ * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
            
        if self._qkv_same_embed_dim:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        else:
            raise NotImplementedError('Not implemented _qkv_diff_embed_dim yet')
            
        if self.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            
        # for test
        if self.finetune_test:
            q = query @ w_q.t() + b_q
            k = key @ w_k.t() + b_k
            v = value @ w_v.t() + b_v
            q = q * self.scale
            q = q.contiguous().view(N, B_ * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.contiguous().view(src_len, B_ * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(src_len,  B_ * self.num_heads, self.head_dim).transpose(0, 1)
            attn = (q @ k.transpose(-2, -1)) # B*Head, N, N
            if attn_mask is not None:
                attn += attn_mask
           
            attn = self.softmax(attn, scale_attn)
            attn = self.attn_drop(attn)
            
            output = (attn @ v) # B*Head, N, C_head
            output = output.transpose(0, 1).contiguous().view(N, B_, C)
            output = output @ self.out_proj.weight.t() + self.out_proj.bias
            # output = self.proj_drop(output)
        else:
            if self.quan_input:
                query, scale_q = self.quan_q(query)
                key, scale_k = self.quan_k(key)
                value, scale_v = self.quan_v(value)
            # quantize linear projection for q/k/v
            w_q, scale_w_q = self.quan_w4q(w_q)
            w_k, scale_w_k = self.quan_w4k(w_k)
            w_v, scale_w_v = self.quan_w4v(w_v)
            if self.in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q = SymmetricQuantFunction.apply(b_q, 16, scale_q * scale_w_q.squeeze()) * scale_q * scale_w_q.squeeze()
                b_k = SymmetricQuantFunction.apply(b_k, 16, scale_k * scale_w_k.squeeze()) * scale_k * scale_w_k.squeeze()
                b_v = SymmetricQuantFunction.apply(b_v, 16, scale_v * scale_w_v.squeeze()) * scale_v * scale_w_v.squeeze()
            q = query @ w_q.t() + b_q
            k = key @ w_k.t() + b_k
            v = value @ w_v.t() + b_v
            # attn
            # scale_m, scale_e = batch_frexp(torch.tensor(self.scale), bit=8)
            # scale_norm = (scale_m / torch.pow(2.0, scale_e)).type(torch.float32)
            # q = q * scale_norm.item()
            q = q * self.scale 
            # reshape to B*Head, N, C_head
            q = q.contiguous().view(N, B_ * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.contiguous().view(src_len, B_ * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(src_len,  B_ * self.num_heads, self.head_dim).transpose(0, 1)
            attn = self.qk_mul(q, k.transpose(-2, -1))
            attn, scale_attn = self.quan_bf_sm(attn)
            if attn_mask is not None:
                attn += attn_mask
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            output = self.attnv_mul(attn, v).transpose(0, 1).contiguous().view(N, B_, C)
            output, scale_output = self.quan_o(output)
            # quantize output projection
            w_out = self.out_proj.weight
            w_out, scale_w_out = self.quan_w4out(w_out)
            b_out = self.out_proj.bias
            b_out = SymmetricQuantFunction.apply(b_out, 16, scale_output * scale_w_out.squeeze()) * scale_output * scale_w_out.squeeze()
            output = output @ w_out.t() + b_out
            output, scale_out = self.quan_out(output)
        return output

        
class QuanConv(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 nbit_w=8, nbit_a=8, stride=1, padding=0, norm=False, act=False,
                 dilation=1, groups=1, bias=True, quan_input=True, input_per_channel=False): # vsq_mode: normal, pw, 3d
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.norm = norm
        self.act = act
        self.w = torch.zeros(self.weight.shape)
        self.b = torch.zeros(self.weight.shape[0])
        self.lsq_w  = LsqQuantizer4weight(
                        bit=self.nbit_w,
                        all_positive=False,
                        per_channel=True,
                        per_channel_num=self.weight.shape[0])
        self.quan_w = self.lsq_w
        self.quan_input = quan_input
        
        if self.quan_input:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                            bit=self.nbit_a,
                            all_positive=False,
                            per_channel=False,
                            per_channel_num=self.weight.shape[1],
                            weight_next=True)
            self.quan_a = self.lsq_a
            
        if self.norm:
            self.bn_weight = torch.nn.parameter.Parameter(torch.ones(out_channels))
            self.bn_bias = torch.nn.parameter.Parameter(torch.zeros(out_channels))
            self.register_buffer('bn_running_mean', torch.zeros(out_channels))
            self.register_buffer('bn_running_var', torch.ones(out_channels))
            self.bn_running_mean = torch.zeros(out_channels)
            self.bn_running_var = torch.ones(out_channels)
            self.momentum = 0.1
        

    # @weak_script_method
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x = self.quan_a(input)
        # quantize input
        x = input
        if self.training and self.norm: 
            output = F.conv2d(x, self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
            mean_bn = output.mean([0, 2,3], keepdim=False).squeeze(0)
            var_bn = output.var([0, 2,3], keepdim=False).squeeze(0)
            output1 = F.batch_norm(output, self.bn_running_mean, self.bn_running_var, weight=self.bn_weight,
                                   bias=self.bn_bias, training=self.training, momentum=0.1, eps=1e-05)
        if not self.training and self.norm:
            mean_bn = self.bn_running_mean
            var_bn = self.bn_running_var
            
        if self.norm:
            tmp = self.bn_weight / torch.sqrt(var_bn + 1e-5)
            w = tmp.view(tmp.size()[0], 1, 1, 1) * self.weight
            b = self.bias
            if self.bias != None:
                b = tmp*(self.bias - mean_bn) + self.bn_bias
            else:
                b = tmp*(0 - mean_bn) + self.bn_bias
        else:
            w = self.weight
            b = self.bias
        
        weight_integer, weight_scaling_factor = self.quan_w(w)
        weight_integer = weight_integer.to(x.device)
        weight_scaling_factor = weight_scaling_factor.to(x.device)
        
        if b != None:
            b = b.to(x.device)
            bias_integer = SymmetricQuantFunction.apply(b, 16, scale_x * weight_scaling_factor.squeeze()) * scale_x * weight_scaling_factor.squeeze()
            bias_integer = bias_integer.to(x.device)
        else:
            bias_integer = None

        # new to export onnx, part1
        # if get_global_idx() == 0:
        #     np.save("npz_logging/input.npy",
        #         (x/scale_x).detach().cpu().numpy()
        #         )  
        
        output2 = F.conv2d(x, weight_integer, bias_integer, self.stride, self.padding, self.dilation, self.groups) 

        if self.training and self.norm:
            output1 = output1 - output2.detach() + output2
            
        #new to export onnx, part2
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()
            if bias_integer != None:
                np.savez("npz_logging/" + str(idx) + "_conv",
                         w=weight_integer.detach().cpu().numpy(),
                         b=bias_integer.detach().cpu().numpy(), 
                         input_scale=scale_x.detach().cpu().numpy(), 
                         weight_scales=weight_scaling_factor.detach().cpu().numpy(), 
                         input=x.detach().cpu().numpy(), 
                         output=output2.detach().cpu().numpy()
                         )
            else:
                np.savez("npz_logging/" + str(idx) + "_conv",
                         w=weight_integer.detach().cpu().numpy(), 
                         input_scale=scale_x.detach().cpu().numpy(), 
                         weight_scales=weight_scaling_factor.detach().cpu().numpy(), 
                         input=x.detach().cpu().numpy(), 
                         output=output2.detach().cpu().numpy()
                         )
            
        return output2
        
class QuanLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, quan_input=True, nbit_w=8, nbit_a=8):
        super(QuanLinear, self).__init__(
            in_features, out_features, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.w = torch.zeros(self.weight.shape)
        self.b = torch.zeros(self.weight.shape[0])
        self.lsq_w  = LsqQuantizer4weight(
                        bit=self.nbit_w,
                        all_positive=False,
                        per_channel=True,
                        per_channel_num=self.weight.shape[0])
        self.quan_w = self.lsq_w
        self.quan_input = quan_input
        if self.quan_input:
            self.lsq_a  = LsqQuantizer4input(
                            bit=self.nbit_a,
                            all_positive=False,
                            per_channel=False)
            self.quan_a = self.lsq_a
            
    def forward(self, x, scale_x=None, scale_for_w=None):
        if self.quan_input:
            x, scale_x = self.quan_a(x)
        # quantize input
        w = self.weight
        b = self.bias
        weight_integer, weight_scaling_factor = self.quan_w(w)
        weight_integer = weight_integer.to(x.device)
        weight_scaling_factor = weight_scaling_factor.to(x.device)
        if b is not None:
            b = b.to(x.device)
            bias_integer = SymmetricQuantFunction.apply(b, 16, scale_x * weight_scaling_factor.squeeze()) * scale_x * weight_scaling_factor.squeeze()
            bias_integer = bias_integer.to(x.device)
        else:
            bias_integer = None
        output = F.linear(x, weight_integer, bias_integer)
        return output
   
    
def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize floating point input tensor to integers with the given scaling factor and zeropoint.

    Parameters:
    ----------
    input: floating point input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    # reshape scale and zeropoint for convolutional weights and activations
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    if inplace:
        input.mul_(1. / scale).add_(zero_point).round_()
        return input
    return torch.round(1. / scale * input + zero_point)


class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        n = 2 ** (k - 1) - 1

        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")

        zero_point = torch.tensor(0.).to(x.device)

        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)

        new_quant_x = torch.clamp(new_quant_x, -n - 1, n)

        ctx.scale = scale
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)

        return grad_output.clone() / scale, None, None, None


class PActFn(Function):
    @staticmethod
    def forward(ctx, x, alpha, k): # k=8
        # if alpha <= 0:
        #     print ("org scale:", alpha, alpha.grad)
        ctx.save_for_backward(x, alpha)
        pow = torch.ceil(torch.log2(alpha**2)) 
        clip_val = torch.pow(2, pow)
    # y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
        scale = (2**(k - 1)) / clip_val
        y = torch.clamp(x + torch.sign(x) * 1e-6, min = -clip_val.item(), max = ((2**(k - 1) - 1) / scale).item())
        y_q = torch.trunc(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        # Backward function, I borrowed code from
        # https://github.com/obilaniu/GradOverride/blob/master/functional.py
        # We get dL / dy_q as a gradient
        # print(current_thread())
        x, alpha, = ctx.saved_tensors
        # Weight gradient is only valid when [0, alpha]
        # Actual gradient for alpha,
        # By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
        # dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range
        dldy_sum = torch.sum(dLdy_q)
        # if dldy_sum.item() != dldy_sum.item():
        #     dLdy_q = torch.zeros(dLdy_q.shape).to(dLdy_q.device)
        #     print ("PACT grad error 1", dLdy_q.shape)
        #     print (dLdy_q)
        #     print ("###############################################")
        #     print (alpha)
        #     print ("###############################################")
        #     print (x)
        #     exit()

        pow = torch.ceil(torch.log2(alpha**2))
        clip_val = torch.pow(2, pow)
        lower_bound = x < -clip_val
        upper_bound = x > clip_val
        # x_range       = 1.0-lower_bound-upper_bound
        x_range = ~(lower_bound |upper_bound)
        grad_alpha = 2 * alpha * torch.sum(dLdy_q * (torch.ge(x, clip_val) | torch.le(x, -clip_val)).float()).view(-1)
        # if grad_alpha.item() != grad_alpha.item():
        #     print ("PACT grad error 2", x.shape)
        #     print (alpha)
        #     print ("###############################################")
        #     print (x)
        #     exit()
        # x_range_sum = torch.sum(x_range.float())
        # if x_range_sum.item() != x_range_sum.item():
        #     print ("PACT grad error 3")
        #     exit()

        # if alpha - grad_alpha <= 0:
        #     grad_alpha = torch.zeros(grad_alpha.shape).to(grad_alpha.device)
        return dLdy_q * x_range.float(), grad_alpha, None

class PACT(nn.Module):
    def __init__(self, num_bits):
        super(PACT, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([2]), requires_grad=True)

    def forward(self, x):

        # pow = torch.ceil(torch.log2(self.clip_val))
        # self.clip_val[0] = torch.pow(2, pow)[0]
        # x = F.relu(x)
        # x = torch.where(x < self.clip_val, x, self.clip_val)
        # x = torch.where(x < self.clip_val, x, self.clip_val)
        # x = torch.where(x > -self.clip_val, x, -self.clip_val)
        # n = float(2 ** (self.num_bits - 1)) / self.clip_val
        # x_forward = torch.round(x * n) / n
        # out = x_forward + x - x.detach()
        out = PActFn.apply(x, self.clip_val, self.num_bits)
        pow = torch.ceil(torch.log2(self.clip_val**2))
        max_out = torch.pow(2, pow)[0]
        scale_a = max_out / float((2**(self.num_bits - 1)))
        return out, scale_a