from prettytable import PrettyTable
import torch.nn as nn
import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)

    return model

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
opt = parser.parse_args()   


model = detect(cfg, opt)
count_parameters(model)

total_params = sum(p.numel() for p in model.parameters())
total_params_m = total_params/1e6
print(f"Total parameters: {total_params_m}")


input_size = ( 384, 640, 3)  # Input size of (3 channels, 256x256 image)
total_flops = 0

# Go through each layer in the model and estimate FLOPs
for layer in model.modules():
    
    if isinstance(layer, nn.Conv2d):
        # print(layer)
        # print('....')
        
        # For convolution layers
        flops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * input_size[1] * input_size[2]
    elif isinstance(layer, nn.Linear):
        # For fully connected layers
        flops = layer.in_features * layer.out_features
    else:
        flops = 0  # For other layers like ReLU, BatchNorm, etc.

    total_flops += flops

# Convert to MFLOPS
total_mflops = total_flops / 1e9
print(f"Total GFLOPS: {total_mflops:.2f} GFLOPS")


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import re
import os
import torch
from torch import nn



macs, params = get_model_complexity_info(model, (3,384, 640), as_strings=True,
                                        print_per_layer_stat=True, verbose=True)
# Extract the numerical value
flops = eval(re.findall(r'([\d.]+)', macs)[0])*2

# Extract the unit
flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
print('Computational complexity: {:<8}'.format(macs))
print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
print('Number of parameters: {:<8}'.format(params))

