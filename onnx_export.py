import argparse
import os
import sys

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from utils.export import export_onnx
from utils.helpers import get_instance
import models
from ruamel.yaml import YAML
from bunch import Bunch



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_path", type=str)
    parser.add_argument("--resolution", type=int, nargs="+", default=224)
    parser.add_argument("--op_set", type=int, default=16)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("-wp", "--weight_path", default="pretrained_weights/DRIVE/checkpoint-epoch40.pth", type=str,
                        help='the path of wetght.pt')

    args = parser.parse_args()

    resolution = (960, 999) # 999x960
    #checkpoint = torch.load(args.weight_path)
    
    
    yaml = YAML(typ='safe', pure=True)
    with open('config_quan.yaml', 'r') as file:
        CFG = Bunch(yaml.load(file))
        
    model = get_instance(models, 'model', CFG)
    #model.load_state_dict(checkpoint['state_dict'])
    
    
    
    
    # 加载预训练模型的状态字典
    pretrained_dict = torch.load(args.weight_path)

    # 移除因为分布式训练而有的 'module.' 前缀
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # 移除 'module.' 前缀
        new_state_dict[k] = v

    # 加载修改后的状态字典到目标模型
    model.load_state_dict(new_state_dict, strict=False)
    

    dummy_input = torch.rand(1, 1, 592, 592) #torch.rand((args.bs, 1, *resolution))
    export_onnx(model, args.export_path, dummy_input, simplify=True, opset=args.op_set)


if __name__ == "__main__":
    main()