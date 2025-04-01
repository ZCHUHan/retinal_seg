import argparse
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch
import os
from ruamel.yaml import YAML
from bunch import Bunch



def main(CFG, data_name, data_path, batch_size, with_val=False, resume=None):
    seed_torch()
    if with_val:
        train_dataset = vessel_dataset(data_path, mode="training", split=0.9)
        val_dataset = vessel_dataset(
                data_path, mode="training", split=0.9, is_val=True)
        val_loader = DataLoader(
            val_dataset, batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    else:
        train_dataset = vessel_dataset(data_path, mode="training")
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    
    model_name = CFG['model']['type']+'_'+CFG['model']['args']['upsample']
    logger.add(os.path.join(CFG.save_dir, model_name, data_name, "log.txt"))
    logger.info(f'\n{model_name}\n')
    logger.info('The patch number of train is %d' % len(train_dataset))
    model = get_instance(models, 'model', CFG)
    logger.info(f'\n{model}\n')
    loss = get_instance(losses, 'loss', CFG)
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader if with_val else None,
        dataset_name=data_name,
        resume=resume  # 
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="/home/lwt/data_pro/vessel/DRIVE", type=str,
                        help='the path of dataset')
    parser.add_argument('-dn', '--dataset_name', default="DRIVE", type=str,
                        help='the name of dataset')
    parser.add_argument('-bs', '--batch_size', default=256,
                        help='batch_size for trianing and validation')
    parser.add_argument("--val", help="split training data for validation",
                        required=False, default=False, action="store_true")
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume training from (default: None)')
    args = parser.parse_args()

    yaml = YAML(typ='safe', pure=True)
    with open('config_quan.yaml', 'r') as file:
        CFG = Bunch(yaml.load(file))
    main(CFG, args.dataset_name, args.dataset_path, args.batch_size, args.val, args.resume)
