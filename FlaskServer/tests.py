import os
import sys
import cv2
import numpy as np
import logging

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

def normalize_and_colorize(depth: np.ndarray, clip_percentile: float = 95) -> np.ndarray:
    """ Normalize and apply colormap to depth map for visualization. """
    max_val = np.percentile(depth, clip_percentile)
    min_val = np.min(depth)

    norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    depth_vis = (norm * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    return color_map

def test_MDE():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Failed to load image from {IMAGE_PATH}")
    
    print("Running Marigold inference...")
    marigold_depth = marigold_inference.estimate_depth(image)
    
    print("Running Metric3D inference...")
    metric3d_depth, confidence = metric3d_inference.estimate_depth(
        version='giant', org_rgb=image, focal_length_px=1000
    )

    # ----- Visualizations -----
    marigold_vis = normalize_and_colorize(marigold_depth)
    metric3d_vis = normalize_and_colorize(metric3d_depth)
    confidence_vis = normalize_and_colorize(confidence)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "marigold_depth_vis.png"), marigold_vis)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "metric3d_depth_vis.png"), metric3d_vis)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "metric3d_confidence_vis.png"), confidence_vis)

    print("Depth visualizations saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    test_MDE()
