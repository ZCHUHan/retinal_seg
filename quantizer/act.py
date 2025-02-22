from typing import Dict, Optional, Type
import torch
import numpy as np
import torch.nn as nn
from models.utils import build_kwargs_from_config
from .lsq import *
from .quant_lsq import * #PACT, get_next_global_idx, get_global_idx
from .GQA_LUT.gqa_lut_op import * #gqa_lut_pwl
        
class Quanhswish(nn.Module):
    def __init__(self, bit=16, quan_a='lsq') -> None:
        super(Quanhswish, self).__init__()
        self.bit = bit
        self.hswish = nn.Hardswish()
        self.pwl_hswish = gqa_lut_pwl(pwl_type='hswish', pwl_dir='quantizer/GQA_LUT/pretrained/hswish_pwl_7.json')
        self.lsq_a = LsqQuantizer4input(
                        bit=bit,
                        all_positive=False,
                        per_channel=False,
                    ) if quan_a == 'lsq' else PACT(num_bits=bit)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_quan, scale_x = self.lsq_a(input)
        act = self.pwl_hswish(x_quan/scale_x, scale_x)
        # x_quan = input
        # act = self.hswish(x_quan) # train w/o pwlq
        if not self.training and get_global_idx() >= 0: #log npz:
            global_idx = get_next_global_idx()
            decimal_scale = torch.tensor([2 ** (-6)], dtype=torch.float32).to(input.device) # 6 decimal bit
            lower_bound = torch.tensor([-32768 * scale_x], dtype=torch.float32).to(input.device)
            np.savez("npz_logging/" + str(global_idx) + "_pwl",
                     input_scale=scale_x.cpu().numpy(), 
                     input=x_quan.cpu().numpy(), 
                     output=act.cpu().numpy(),
                     slopes=(self.pwl_hswish.slopes).cpu().numpy(), 
                     intercepts=(self.pwl_hswish.intercepts * scale_x).cpu().numpy(), 
                     change_pts=torch.cat((lower_bound, self.pwl_hswish.breakpoints * scale_x)).cpu().numpy(),
                     slopes_scale=decimal_scale.cpu().numpy(), 
                     intercepts_scale=decimal_scale.cpu().numpy()
                     )
        return act   
    
       
# register activation function here
REGISTERED_ACT_DICT: Dict[str, Type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "quanhswish": Quanhswish(),
    "hswish": nn.Hardswish,
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        if name=="quanhswish":# or name=="hswish":
            return act_cls
        else:
            return act_cls(**args)
    else:
        return None
