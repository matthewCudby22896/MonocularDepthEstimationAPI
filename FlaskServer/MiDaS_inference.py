import torch
import logging
import cv2
import numpy as np

from FlaskServer.types import ImageNP
from MiDaS.midas.model_loader import load_model
from MiDaS.run import process

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./MiDaS/weights/dpt_beit_large_512.pt"
MODEL_TYPE = "dpt_beit_large_512"
OPTIMIZE = False

model = transform = net_w = net_h = None

def monocular_depth_estimation(image_bgr: ImageNP) -> np.ndarray:
    global model, transform, net_w, net_h

    logger.info("MiDas")
    
    if model is None:
        model, transform, net_w, net_h = load_model(
            device,
            MODEL_PATH,
            MODEL_TYPE,
            optimize=OPTIMIZE,
            height=512,
            square=False
        )
            
    # Convert BGR to RGB if needed
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255

    # Prepare input for the model
    sample = transform({'image': image_rgb})["image"]

    # Predict depth
    with torch.no_grad():
        pred_disparity : np.ndarray = process(
            device,
            model,
            MODEL_TYPE,
            sample,
            (net_w, net_h),
            image_rgb.shape[1::-1],
            optimize=OPTIMIZE,
            use_camera=False
        )
    
    # Avoid messing with valid values
    valid_mask = pred_disparity > 0

    # Safely invert only valid points
    pred_depth = np.zeros_like(pred_disparity)
    pred_depth[valid_mask] = 1.0 / pred_disparity[valid_mask]

    # Assign farthest possible value to invalid (e.g. sky)
    max_depth = pred_depth[valid_mask].max()
    pred_depth[~valid_mask] = max_depth
    
    return pred_depth.astype(np.float32)
    

