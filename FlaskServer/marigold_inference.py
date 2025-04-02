import logging
import os
import sys
import cv2
import numpy as np
import torch
import io


from metric3d_inference import Image
from Marigold.marigold.marigold_pipeline import MarigoldPipeline

CHECKPOINT_PATH = "./Marigold/checkpoint/marigold-lcm-v1-0"
DENOISING_STEPS = 4 # (1-4 for LCM)
ENSEMBLE_SIZE = 5   # (Ensemble size) 
RESAMPLE_METHOD = 'bilinear'
SEED = 1
BATCH_SIZE = 0
MATCH_INPUT_RES = True
PROCESSING_RES = 0 # zero for using input image resolution

def estimate_depth(image : Image,
                   checkpoint_path : str = CHECKPOINT_PATH,
                   denoise_steps : int = DENOISING_STEPS,
                   ensemble_size : int = ENSEMBLE_SIZE,
                   processing_res : int = PROCESSING_RES,
                   seed : int = SEED) -> np.array:
    """
    """
    assert isinstance(image, np.ndarray), "Input image must be a numpy array"
    assert image.ndim == 3 and image.shape[2] == 3, "Expected RGB image (H, W, 3)"

    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")
    
    dtype = torch.float32
    variant = None
    
    pipe: MarigoldPipeline = MarigoldPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )
    
    pipe = pipe.to(device)
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )
    
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}; "
    )
    
    with torch.no_grad():
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        image = cv2_image_to_pil_file_buffer(image)
        pipe_out = pipe(
                image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=True,
                batch_size=1,
                show_progress_bar=True,
                resample_method=RESAMPLE_METHOD,
                generator=generator,
            )
    
        depth_pred: np.ndarray = pipe_out.depth_np
        
    return depth_pred


def cv2_image_to_pil_file_buffer(image: np.ndarray, format: str = "PNG") -> io.BytesIO:
    # Convert OpenCV BGR image to RGB for compatibility with PIL encoding expectations
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Encode image to chosen format (PNG default is lossless)
    success, encoded_img = cv2.imencode(f'.{format.lower()}', rgb_image)
    if not success:
        raise ValueError("Failed to encode image")

    # Wrap bytes in a BytesIO stream, as expected by PIL/open-style APIs
    return io.BytesIO(encoded_img.tobytes())


