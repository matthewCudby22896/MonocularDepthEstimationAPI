import os
import os.path as osp
import cv2
import time
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
import argparse
import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from datetime import timedelta
import random
import numpy as np
from mono.utils.logger import setup_logger
import glob
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data
from mono.utils.do_test import get_prediction, resize_for_input, build_camera_model

Image = np.ndarray

models = {}

vit_size=(616,1064)

config_paths = {
    'small' : 'mono/configs/HourglassDecoder/vit.raft5.small.py' ,
    'large' : 'mono/configs/HourglassDecoder/vit.raft5.large.py' ,
    'giant' : 'mono/configs/HourglassDecoder/vit.raft5.giant2.py' ,
}

weight_paths = {
    'small' : '',
    'large' : '',
    'giant' : 'weight/metric_depth_vit_giant2_800k.pth'
}

VIT_settings = dict(
    canonical_space = dict(
        # img_size=(540, 960),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.1, 200),
    crop_size = (616, 1064),  # %28 = 0
     clip_depth_range=(0.1, 200),
    vit_size=(616,1064)
) 

def get_configured_model(model_version : str) -> torch.nn.Module:
    if model_version in models:
        return models[model_version]
    
    cfg_pth = config_paths.get(model_version, None)
    if cfg_pth is None:
        raise Exception(f'no config path for {model_version=}')
    
    cfg : Config = Config.fromfile(cfg_pth)
    
    weights_path = weight_paths.get(model_version, None)
    if weights_path is None:
        raise Exception(f'no config path for {model_version=}')
    
    cfg.batch_size = 1
    cfg.distributed = False
    
    model = get_configured_monodepth_model(cfg, )
    
    checkpoint = torch.load(weights_path, map_location='cuda')
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])  # If using a full checkpoint
    else:
        model.load_state_dict(checkpoint)  # If directly loading state dict

    models[model_version] = model
    
    return models[model_version]

def run_inference(img : Image, focal_length : float, model_version : str) -> np.ndarray:
    model : torch.nn.Module = get_configured_model(model_version)
    model.eval()
    
    # Assume
    p_x, p_y = img.shape[1] // 2, img.shape // 2
    
    intrinsic = [focal_length, focal_length, p_x, p_y]
    
    to_canonical_ratio = focal_length / VIT_settings['canonical_space']['focal_length']
    img, cam_model, _, _ = resize_for_input(img,
                     output_shape=VIT_settings['crop_size'],
                     intrinsic=intrinsic,
                     canonical_shape=VIT_settings['vit_size'],
                     to_canonical_ratio=to_canonical_ratio)
        
    get_prediction(model, input, cam_model)

    
    
    
    
    
    

