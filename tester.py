import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta
import os
from thop import profile
from torchstat import stat

def reshape_img(image, y, x):
    out = np.zeros([image.shape[0], image.shape[1], y, x], dtype=np.float32)
    out[:, :, 0:image.shape[2], 0:image.shape[3]] = image[:, :, 0:image.shape[2],
                                                    0:image.shape[3]]
    return out

class Tester(Trainer):
    def __init__(self, model, loss, CFG, checkpoint, test_loader, dataset_path, show=False, save_path=None, patch_size=None, stride=None):
        # super(Trainer, self).__init__()
        self.loss = loss
        self.CFG = CFG
        self.test_loader = test_loader
        self.model = nn.DataParallel(model.cuda())
        self.dataset_path = dataset_path
        self.show = show
        self.model.load_state_dict(checkpoint['state_dict'])
        if patch_size is not None:
            self.save_path = os.path.join(save_path, dataset_path.split("/")[-1]+"_patch"+str(patch_size)+"_stride"+str(stride))
        else:
            self.save_path = os.path.join(save_path, dataset_path.split("/")[-1])
        self.patch_size = patch_size
        self.stride = stride
        if self.show:
            dir_exists(self.save_path)
            remove_files(self.save_path)
        cudnn.benchmark = True
        
    
    def count(self, model, patch_size=None):
        if self.dataset_path.endswith("DRIVE"):
            shape = 592
        elif self.dataset_path.endswith("CHASEDB1"):
            shape = 1008
        elif self.dataset_path.endswith("DCA1"):
            shape = 320
        elif self.dataset_path.endswith("STARE"):
            shape = 704
        if patch_size is not None:
            shape = patch_size
        input = torch.randn(1, 1, shape, shape).cuda()
        macs, params = profile(model, inputs=(input, ))
        model = model.cpu()
        stat(model, (1, shape, shape))
        print('macs:', f"{macs / 1000000000:.2f}G")
        print('params:', f"{params / 1000000:.2f}M")
        print('shape:', shape)

    def test(self):
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                # print('img.shape:', img.shape)
                pre = self.model(img)
                loss = self.loss(pre, gt)
                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)
                
                if self.dataset_path.endswith("DRIVE"):
                    H, W = 584, 565
                elif self.dataset_path.endswith("CHASEDB1"):
                    H, W = 960, 999
                elif self.dataset_path.endswith("DCA1"):
                    H, W = 300, 300
                elif self.dataset_path.endswith("STARE"):
                    H, W = 605, 700

                if not self.dataset_path.endswith("CHUAC"):
                    img = TF.crop(img, 0, 0, H, W)
                    gt = TF.crop(gt, 0, 0, H, W)
                    pre = TF.crop(pre, 0, 0, H, W)
                img = img[0,0,...]
                gt = gt[0,0,...]
                pre = pre[0,0,...]
                if self.show:
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                    cv2.imwrite(
                        os.path.join(self.save_path, "img"+str(i)+".png"), np.uint8(img.cpu().numpy()*255))
                    cv2.imwrite(
                        os.path.join(self.save_path, "gt"+str(i)+".png"), np.uint8(gt.cpu().numpy()*255))
                    cv2.imwrite(
                        os.path.join(self.save_path, "pre"+str(i)+".png"), np.uint8(predict*255))
                    cv2.imwrite(
                        os.path.join(self.save_path, "pre_b"+str(i)+".png"), np.uint8(predict_b*255))

                if self.CFG.DTI:
                    pre_DTI = double_threshold_iteration(
                        i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    self._metrics_update(
                        *get_metrics(pre, gt, predict_b=pre_DTI).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(pre_DTI, gt))
                else:
                    self._metrics_update(
                        *get_metrics(pre, gt, self.CFG.threshold).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(
                            pre, gt, threshold=self.CFG.threshold))
                tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                tic = time.time()
        logger.add(os.path.join(self.save_path, "log.txt"))
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')
        logger.info(f'     loss:  {self.total_loss.average}')
        if self.CFG.CCC:
            logger.info(f'     CCC:  {self.CCC.average}')
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')
    
    def test_sliwindow(self):
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():
            for idx, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.numpy()
                
                #sliding window
                y, x = img.shape[2], img.shape[3]
                ind = (max(y, x) // self.patch_size + 1) * self.patch_size
                print('ori--img.shape:', img.shape)
                img = reshape_img(img, ind, ind)
                print('res--img.shape:', img.shape)
                print('ind:', ind)
                predict = np.zeros([1, 1, ind, ind], dtype=np.float32)
                n_map = np.zeros([1, 1, ind, ind], dtype=np.float32)
                
                """
                Our prediction is carried out using sliding patches, 
                and for each patch a corresponding result is predicted, 
                and for the part where the patches overlap, 
                we use weight <map_kernel> balance, 
                and we agree that the closer to the center of the patch, the higher the weight
                """
                shape = (self.patch_size, self.patch_size)
                a = np.zeros(shape=shape)
                a = np.where(a == 0)
                epsilon = 1e-10
                map_kernal = 1 / ((a[0] - shape[0] // 2)**4 +
                                (a[1] - shape[1] // 2)**4 + epsilon)
                map_kernal = np.reshape(map_kernal, newshape=(1,1, ) + shape)
                
                # stride_x = shape[0] // 2
                # stride_y = shape[1] // 2
                stride_x = self.stride
                stride_y = self.stride
                window_count = 0
                for i in range(y // stride_x):
                    for j in range(x // stride_y):
                        # current window
                        img_i = img[:, :, i * stride_x:i * stride_x + shape[1],
                                    j * stride_y:j * stride_y + shape[0], ] 
                        img_i = torch.from_numpy(img_i)
                        img_i = img_i.cuda()
                        window_count += 1
                        output = self.model(img_i)
                        output = output.data.cpu().numpy()
                        
                        predict[:, :, i * stride_x:i * stride_x + shape[0],
                                j * stride_y:j * stride_y + shape[1], ] += (output *
                                                                            map_kernal)
                        n_map[:, :, i * stride_x:i * stride_x + shape[0],
                            j * stride_y:j * stride_y + shape[1], ] += map_kernal
                    
                    # right edge    
                    img_i = img[:, :, i * stride_x:i * stride_x + shape[1],
                            y - shape[0]:y]
                    img_i = torch.from_numpy(img_i)
                    img_i = img_i.cuda()
                    window_count += 1
                    output = self.model(img_i)
                    output = output.data.cpu().numpy()
                    

                    predict[:, :, i * stride_x:i * stride_x + shape[0],
                            y - shape[0]:y] += (output * map_kernal)
                    n_map[:, :, i * stride_x:i * stride_x + shape[0],
                        y - shape[0]:y] += map_kernal
                
                # bottom edge
                for j in range(x // stride_y - 1):
                    img_i = img[:, :, x - shape[1]:x,
                                    j * stride_y:j * stride_y + shape[1]]

                    img_i = torch.from_numpy(img_i)
                    img_i = img_i.cuda()
                    window_count += 1
                    output = self.model(img_i)
                    output = output.data.cpu().numpy()
                    

                    predict[:, :, x - shape[1]:x, j * stride_y:j * stride_y +
                            shape[1]] += (output * map_kernal)
                    n_map[:, :, x - shape[1]:x,
                        j * stride_y:j * stride_y + shape[1]] += map_kernal
                
                # Bottom right corner
                img_i = img[:, :, x - shape[1]:x, y - shape[0]:y]
                img_i = torch.from_numpy(img_i)
                if torch.cuda.is_available():
                    img_i = img_i.cuda()
                window_count += 1
                output = self.model(img_i)
                output = output.data.cpu().numpy()
                
               
                predict[:, :, x - shape[1]:x, y - shape[0]:y] += output * map_kernal
                n_map[:, :, x - shape[1]:x, y - shape[0]:y] += map_kernal
                
                # output = predict / n_map
                output = predict / (n_map + epsilon)
                output = output.astype(dtype=np.float32)
                output_final = np.zeros([1, 1, y, x], dtype=np.float32)
                output_final[:, :, 0:y, 0:x] = output[:, :, 0:y, 0:x]
                print('sliding window num: ', window_count)
                
                img = torch.from_numpy(img).cuda()
                pre = torch.from_numpy(output_final).cuda()
                gt = gt.cuda()
                loss = self.loss(pre, gt)
                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)
                
                if self.dataset_path.endswith("DRIVE"):
                    H, W = 584, 565
                elif self.dataset_path.endswith("CHASEDB1"):
                    H, W = 960, 999
                elif self.dataset_path.endswith("DCA1"):
                    H, W = 300, 300
                elif self.dataset_path.endswith("STARE"):
                    H, W = 605, 700
                
                if not self.dataset_path.endswith("CHUAC"):
                    img = TF.crop(img, 0, 0, H, W)
                    gt = TF.crop(gt, 0, 0, H, W)
                    pre = TF.crop(pre, 0, 0, H, W)
                img = img[0,0,...]
                gt = gt[0,0,...]
                pre = pre[0,0,...]
                if self.show:
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                    cv2.imwrite(
                        os.path.join(self.save_path, "img"+str(idx)+".png"), np.uint8(img.cpu().numpy()*255))
                    cv2.imwrite(
                        os.path.join(self.save_path, "gt"+str(idx)+".png"), np.uint8(gt.cpu().numpy()*255))
                    cv2.imwrite(
                        os.path.join(self.save_path, "pre"+str(idx)+".png"), np.uint8(predict*255))
                    cv2.imwrite(
                        os.path.join(self.save_path, "pre_b"+str(idx)+".png"), np.uint8(predict_b*255))

                if self.CFG.DTI:
                    pre_DTI = double_threshold_iteration(
                        idx, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    self._metrics_update(
                        *get_metrics(pre, gt, predict_b=pre_DTI).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(pre_DTI, gt))
                else:
                    self._metrics_update(
                        *get_metrics(pre, gt, self.CFG.threshold).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(
                            pre, gt, threshold=self.CFG.threshold))
                tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        idx, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                tic = time.time()
        logger.add(os.path.join(self.save_path, "log.txt"))
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')
        logger.info(f'     loss:  {self.total_loss.average}')
        if self.CFG.CCC:
            logger.info(f'     CCC:  {self.CCC.average}')
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')
                
        