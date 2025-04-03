import os
import sys
import cv2
import numpy as np
import logging
import time

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./Metric3D"))
sys.path.append(os.path.abspath("./Marigold"))

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

import marigold_inference
import metric3d_inference

OUTPUT_DIR = "FlaskServer/test_outputs"
IMAGE_PATH = "FlaskServer/test_images/image.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalise_and_colourise(depth: np.ndarray, clip_percentile: float = 95) -> np.ndarray:
    """ Normalise and apply colourmap to depth map for visualisation. """
    max_val = np.percentile(depth, clip_percentile)
    min_val = np.min(depth)

    norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    depth_vis = (norm * 255).astype(np.uint8)
    colour_map = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    return colour_map

def test_MDE():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Failed to load image from {IMAGE_PATH}")
    
    logger.info(f"Running Marigold inference...")
    s = time.time()
    marigold_depth = marigold_inference.estimate_depth(image)
    marigold_time = time.time() - s
    logger.info(f"Marigold: image shape: {image.shape}, depth map shape: {marigold_depth.shape}")
    
    assert marigold_depth.shape[0] == image.shape[0] and marigold_depth.shape[1] == marigold_depth.shape[1]

    logger.info("Running Metric3D inference...")
    s = time.time()
    metric3d_depth, confidence = metric3d_inference.estimate_depth(
        version='giant', org_rgb=image, focal_length_px=1000
    )
    metric3d_time = time.time()
    logger.info(f"Metric3D: image shape: {image.shape}, depth map shape: {metric3d_depth.shape}")
    
    assert metric3d_depth.shape[0] == metric3d_depth.shape[0] and metric3d_depth.shape[1] == metric3d_depth.shape[1]
    assert confidence.shape[0] == confidence.shape[0] and confidence.shape[1] == confidence.shape[1]
    
    # ----- Visualisations -----
    marigold_vis = normalise_and_colourise(marigold_depth)
    metric3d_vis = normalise_and_colourise(metric3d_depth)
    confidence_vis = normalise_and_colourise(confidence)
    
    logger.info(f"Inference Times:\n\tMarigold: {marigold_time}\n\tMetric3D: {metric3d_time}")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "marigold_depth_vis.png"), marigold_vis)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "metric3d_depth_vis.png"), metric3d_vis)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "metric3d_confidence_vis.png"), confidence_vis)

    logger.info(f"Depth visualisations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    test_MDE()
