import argparse
import torch
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from tester import Tester
from utils import losses
from utils.helpers import get_instance
from ruamel.yaml import YAML
from bunch import Bunch


def main(data_path, weight_path, CFG, show, save_path, patch_size=None, stride=None):
    checkpoint = torch.load(weight_path)
    CFG_ck = checkpoint['config']
    test_dataset = vessel_dataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1,
                             shuffle=False,  num_workers=16, pin_memory=True)
    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG_ck)
    test = Tester(model, loss, CFG, checkpoint, test_loader, data_path, show, save_path, patch_size, stride)
    if patch_size is not None:
        test.test_sliwindow()
        # test.count(model, patch_size)
    else:
        test.test()
        # test.count(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default="/home/lwt/data_pro/vessel/DRIVE", type=str,
                        help="the path of dataset")
    parser.add_argument("-wp", "--wetght_path", default="pretrained_weights/DRIVE/checkpoint-epoch40.pth", type=str,
                        help='the path of wetght.pt')
    parser.add_argument("--show", help="save predict image",
                        required=False, default=False, action="store_true")
    parser.add_argument("--save_path", help="save path",
                        default="results", action="store_true")
    parser.add_argument("--sliding", help="sliding window for testing",
                        required=False, default=False, action="store_true")
    parser.add_argument('-ps', '--patch_size', default=48, type=int,
                        help='patch_size for sliding window')
    parser.add_argument('--stride', default=44, type=int,
                        help='stride_size for sliding window')
    args = parser.parse_args()
    yaml = YAML(typ='safe', pure=True)
    with open('config_quan.yaml', 'r') as file:
        CFG = Bunch(yaml.load(file))
    if args.sliding:
        main(args.dataset_path, args.wetght_path, CFG, args.show, args.save_path, patch_size=args.patch_size, stride=args.stride)
    else:
        main(args.dataset_path, args.wetght_path, CFG, args.show, args.save_path)
