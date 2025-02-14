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
    parser.add_argument("--bs", type=int, default=2)

    args = parser.parse_args()

    resolution = (960, 999) # 999x960
    
    yaml = YAML(typ='safe', pure=True)
    with open('config_quan.yaml', 'r') as file:
        CFG = Bunch(yaml.load(file))
    model = get_instance(models, 'model', CFG)

    dummy_input = torch.rand(1, 1, 1008, 1008) #torch.rand((args.bs, 1, *resolution))
    export_onnx(model, args.export_path, dummy_input, simplify=True, opset=args.op_set)


if __name__ == "__main__":
    main()