import os
import sys
import cv2
import numpy as np
import logging
import time

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./Metric3D"))
sys.path.append(os.path.abspath("./Marigold"))
sys.path.append(os.path.abspath("./MiDaS"))


logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

import FlaskServer.Marigold_inference as Marigold_inference
import FlaskServer.Metric3D_inference as Metric3D_inference
import FlaskServer.MiDaS_inference as MiDaS_inference

OUTPUT_DIR = "FlaskServer/test_outputs"
IMAGE_PATH = "FlaskServer/test_images/image.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualise_depth_map(depth: np.ndarray, clip_percentile: float = 95) -> np.ndarray:
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

    marigold_time = metric3d_time = midas_time = 0

    # ----- MARIGOLD -----
    try:
        logger.info(f"Running Marigold inference...")
        s = time.time()
        marigold_depth = Marigold_inference.monocular_depth_estimation(image)
        marigold_time = time.time() - s
        logger.info(f"[Marigold] Inference time: {marigold_time:.2f}s")

        logger.info(f"Marigold: image shape: {image.shape}, depth shape: {marigold_depth.shape}")
        marigold_vis = visualise_depth_map(marigold_depth)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "marigold_depth_vis.png"), marigold_vis)

        logger.info(f"[Marigold] Min: {marigold_depth.min():.2f}, Max: {marigold_depth.max():.2f}, Mean: {marigold_depth.mean():.2f}")
    except Exception as e:
        logger.error(f"Marigold failed: {e}")

    # ----- METRIC3D -----
    try:
        logger.info("Running Metric3D inference...")
        s = time.time()
        metric3d_depth, confidence = Metric3D_inference.monocular_depth_estimation(
            version='giant', org_rgb=image, focal_length_px=1000
        )
        metric3d_time = time.time() - s
        logger.info(f"[Metric3D] Inference time: {metric3d_time:.2f}s")

        logger.info(f"Metric3D: depth shape: {metric3d_depth.shape}, confidence shape: {confidence.shape}")
        metric3d_vis = visualise_depth_map(metric3d_depth)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "metric3d_depth_vis.png"), metric3d_vis)

        logger.info(f"[Metric3D] Depth min/max/mean: {metric3d_depth.min():.2f}/{metric3d_depth.max():.2f}/{metric3d_depth.mean():.2f}")
        logger.info(f"[Metric3D] Confidence min/max/mean: {confidence.min():.2f}/{confidence.max():.2f}/{confidence.mean():.2f}")
    except Exception as e:
        logger.error(f"Metric3D failed: {e}")

    # ----- MIDAS -----
    try:
        logger.info("Running MiDaS inference...")
        s = time.time()
        midas_depth = MiDaS_inference.monocular_depth_estimation(image)
        midas_time = time.time() - s
        logger.info(f"[MiDaS] Inference time: {midas_time:.2f}s")

        logger.info(f"MiDaS: depth shape: {midas_depth.shape}")
        midas_vis = visualise_depth_map(midas_depth)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "midas_depth_vis.png"), midas_vis)

        logger.info(f"[MiDaS] Min: {midas_depth.min():.2f}, Max: {midas_depth.max():.2f}, Mean: {midas_depth.mean():.2f}")
    except Exception as e:
        logger.error(f"MiDaS failed: {e}")

    logger.info("\n--- Inference Time Summary ---")
    logger.info(f"Marigold: {marigold_time:.2f}s\nMetric3D: {metric3d_time:.2f}s\nMiDaS: {midas_time:.2f}s")
    logger.info("All available visualisations and depth maps saved to: %s", OUTPUT_DIR)
    
if __name__ == "__main__":
    test_MDE()