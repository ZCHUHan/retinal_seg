import torch
import numpy as np
import decimal
from decimal import Decimal

global_idx = 0

def get_next_global_idx():
    global global_idx
    global_idx = global_idx + 1
    return global_idx

def reset_global_idx(a):
    global global_idx
    global_idx = a

def get_global_idx():
    global global_idx
    return global_idx

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

    output_m, output_e = np.frexp(inputs.cpu().numpy())

    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(m * (2 ** bit)).quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = 1.0*bit - output_e

    return torch.from_numpy(output_m).to(inputs.device).view(shape_of_input), \
           torch.from_numpy(output_e).to(inputs.device).view(shape_of_input)

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

def symmetric_linear_quantization_params(num_bits,
                                         saturation_min,
                                         saturation_max,
                                         per_channel=False):
    """
    Compute the scaling factor and zeropoint with the given quantization range for symmetric quantization.

    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    per_channel: if True, calculate the scaling factor per channel.
    """

    # these computation do not require any gradients, to enforce this, we use torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        scale1, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
        if per_channel:
            # scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale1, min=1e-8) / n
        else:
            scale = torch.max(scale1)
            scale = torch.ones(scale1.shape).to(saturation_min.device) * torch.clamp(scale, min=1e-8) / n
    if torch.sum(torch.isnan(scale.flatten())) == 0:
        scale_m, scale_e = batch_frexp(scale, bit=8)
        scale = (scale_m / torch.pow(2, scale_e)).type(torch.float32)
    return scale

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
        