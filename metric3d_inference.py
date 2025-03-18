
import torch
import numpy as np
import cv2
import time
import logging

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.model.monodepth_model import get_monodepth_model

Image = np.ndarray


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
    'small': "weight/metric_depth_vit_small_800k.pth",
    'large': "weight/metric_depth_vit_large_800k.pth",
    'giant': "weight/metric_depth_vit_giant2_800k.pth",
}

MODEL_CFG = {
    'small': "mono/configs/HourglassDecoder/vit.raft5.small.py",
    'large': "mono/configs/HourglassDecoder/vit.raft5.large.py",
    'giant': "mono/configs/HourglassDecoder/vit.raft5.giant2.py",
}

TORCH_HUB_USER = 'yvanyin/metric3d'
vit_input_size = (616, 1064)

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
        model = get_monodepth_model(cfg)
        
        # Load checkpoint
        weights_path = MODEL_WEIGHTS[version]
        checkpoint = torch.load(weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # Move to GPU and set to eval mode
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        models[version] = model
    
    return models[version]


def estimate_depth(version : str, org_rgb : Image, focal_length_px : float) -> np.ndarray:
    model : torch.nn.Module = get_model(version)
    
    focal_length_px = float(focal_length_px)
        
    logger.info(f"{org_rgb.shape=}")
    
    h, w = org_rgb.shape[:2]
    intrinsic = [focal_length_px, focal_length_px, w // 2, h // 2] # (f_x, f_y, px, py)
    
    # Rescale the image to fit the expected input size
    h, w = org_rgb.shape[:2]
    scale = min(vit_input_size[0] / h, vit_input_size[1] / w)
    rgb = cv2.resize(org_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    
    # Scale the camera intrinsics, in the same way
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    
    # Add padding s.t. the input exactly fits the expect input size
    h, w = rgb.shape[:2]
    pad_h, pad_w = vit_input_size[0] - h, vit_input_size[1] - w
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
        pred_depth, confidence, output_dict = model.inference({'input' : rgb})
        e = time.time()
        logger.info(f"Model inference took {e - s} seconds\n\t{version=}\n\t{focal_length_px=}")
        
    # TODO: Remove debugging
    confidence = confidence.squeeze()
    confidence = confidence[pad_info[0] : confidence.shape[0] - pad_info[1], pad_info[2] : confidence.shape[1] - pad_info[3]]
    confidence = torch.nn.functional.interpolate(confidence[None, None, :, :], org_rgb.shape[:2], mode='bilinear').squeeze()
    confidence = confidence.numpy()
    
    print("confidence:")
    print(f"{confidence.shape}")
    print(confidence)
    print(f"max val : {np.max(confidence)}")
    print(f"min val : {np.min(confidence)}")
    # NOTE: End debugging
    
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
    
    pred_depth_np = pred_depth.numpy()
    
    assert pred_depth_np.shape == org_rgb.shape[0:2]
    
    return pred_depth_np

