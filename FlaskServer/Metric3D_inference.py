
import torch
import numpy as np
import cv2
import time
import logging

from mmengine.config import Config

from Metric3D.mono.model.monodepth_model import get_configured_monodepth_model
from Metric3D.mono.utils.running import load_ckpt
from FlaskServer.types import ImageNP

logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    raise Exception(f"cuda not available!")

PADDING_CLR = [123.675, 116.28, 103.53]

MODEL_VERSIONS = {
    'small': 'metric3d_vit_small',
    'large': 'metric3d_vit_large',
    'giant': 'metric3d_vit_giant2',
}

MODEL_WEIGHTS = {
    'small': "./Metric3D/weight/metric_depth_vit_small_800k.pth",
    'large': "./Metric3D/weight/metric_depth_vit_large_800k.pth",
    'giant': "./Metric3D/weight/metric_depth_vit_giant2_800k.pth",
}

MODEL_CFG = {
    'small': "./Metric3D/mono/configs/HourglassDecoder/vit.raft5.small.py",
    'large': "./Metric3D/mono/configs/HourglassDecoder/vit.raft5.large.py",
    'giant': "./Metric3D/mono/configs/HourglassDecoder/vit.raft5.giant2.py",
}

VIT_INPUT_SIZE = (616, 1064)

models = {}

if not torch.cuda.is_available():
    raise Exception("CUDA not available!")

def get_model(version: str):
    if version not in MODEL_VERSIONS:
        raise ValueError(f"Unknown version: {version}")
    
    if version not in models:
        logger.info(f"Loading {version} model...")
        
        # Load configuration
        cfg = Config.fromfile(MODEL_CFG[version])
        
        # Initialize model
        model = get_configured_monodepth_model(cfg, )
        
        model = torch.nn.DataParallel(model).cuda()
        
        weights_path = MODEL_WEIGHTS[version]
        model, _,  _, _ = load_ckpt(weights_path, model, strict_match=False)
        model.eval()
        
        models[version] = model
    
    return models[version]

def monocular_depth_estimation(version : str, org_rgb : ImageNP, focal_length_px : float) -> np.ndarray:
    model : torch.nn.Module = get_model(version)
    
    focal_length_px = float(focal_length_px)
        
    h, w = org_rgb.shape[:2]
    intrinsic = [focal_length_px, focal_length_px, w // 2, h // 2] # (f_x, f_y, px, py)
    
    # Rescale the image to fit the expected input size
    h, w = org_rgb.shape[:2]
    scale = min(VIT_INPUT_SIZE[0] / h, VIT_INPUT_SIZE[1] / w)
    rgb = cv2.resize(org_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    
    # Scale the camera intrinsics, in the same way
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    
    # Add padding s.t. the input exactly fits the expect input size
    h, w = rgb.shape[:2]
    pad_h, pad_w = VIT_INPUT_SIZE[0] - h, VIT_INPUT_SIZE[1] - w
    pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=PADDING_CLR)
    
    # normalise 
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()
    
    ###################### canonical camera space ######################
    model.cuda().eval()
    with torch.no_grad():
        s = time.time()
        pred_depth, confidence, output_dict = model.module.inference({'input' : rgb})
        e = time.time()
        logger.info(f"Model inference took {e - s} seconds\n\t{version=}\n\t{focal_length_px=}")
        
    confidence = confidence.squeeze()
    confidence = confidence[pad_info[0] : confidence.shape[0] - pad_info[1], pad_info[2] : confidence.shape[1] - pad_info[3]]
    confidence = torch.nn.functional.interpolate(confidence[None, None, :, :], org_rgb.shape[:2], mode='bilinear').squeeze()
    confidence_np = confidence.cpu().numpy()
    
    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], org_rgb.shape[:2], mode='bilinear').squeeze()
    
    ###################### canonical camera space ######################

    #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)
    
    pred_depth_np = pred_depth.cpu().numpy()
    
    assert pred_depth_np.shape == org_rgb.shape[0:2]
    
    return pred_depth_np.astype(np.float32), confidence_np.astype(np.float32)

