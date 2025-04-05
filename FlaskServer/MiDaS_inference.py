import torch
import logging
import cv2
import numpy as np

from MiDaS.midas.model_loader import load_model
from MiDaS.run import process

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./MiDaS/weights/dpt_beit_large_512.pt"
MODEL_TYPE = "dpt_beit_large_512"
OPTIMIZE = False

def monocular_depth_estimation(image_np: np.ndarray) -> np.ndarray:
    logger.info("MiDas")
    
    # Load MiDaS model
    model, transform, net_w, net_h = load_model(
        device,
        MODEL_PATH,
        MODEL_TYPE,
        optimize=OPTIMIZE,
        height=512,
        square=False
    )

    # Convert BGR to RGB if needed
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) / 255

    # Prepare input for the model
    sample = transform({'image': image_rgb})["image"]

    # Predict depth
    with torch.no_grad():
        prediction = process(
            device,
            model,
            MODEL_TYPE,
            sample,
            (net_w, net_h),
            image_rgb.shape[1::-1],
            OPTIMIZE,
            use_camera=False
        )
    
    logger.info("Prediction Complete")
    
    print(prediction)

    return prediction
