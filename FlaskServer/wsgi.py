import os
import sys
from flask import Flask, request, jsonify, send_file
import io
import numpy as np
import cv2
from torch import Tuple

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./Metric3D"))
sys.path.append(os.path.abspath("./Marigold"))
sys.path.append(os.path.abspath("./FlaskServer"))

from FlaskServer import Marigold_inference
from FlaskServer import Metric3D_inference
from FlaskServer import MiDaS_inference

from FlaskServer.types import ImageNP

from FlaskServer.constants import MODEL_OPTIONS, METRIC_3D, MARIGOLD, MIDAS, logger, DEFAULT_DENOISING_STEPS, DEFAULT_ENSEMBLE_SIZE


app = Flask(__name__)

# --------------------- Route ---------------------
@app.route("/inference/", methods=['POST'])
def run_inference():
    """
    Run MDE inference on an uploaded image.
    Run with (from project root):
        flask --app FlaskServer.wsgi run --host=0.0.0.0
        
    """

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
                
    # Log request parameters
    logger.info("Request Received with parameters:")
    json_data = request.form.to_dict()
    for key, value in json_data.items():
        logger.info(f"  {key}: {value}")

    model, version, focal_length, denoising_steps, ensemble_size = (json_data.get('model'), 
                                                                    json_data.get('version'),
                                                                    json_data.get('focal_length'),
                                                                    json_data.get('denoising_steps'),
                                                                    json_data.get('ensemble_size'))

    # Validate model & version
    if model not in MODEL_OPTIONS:
        err = f"Unexpected model: {model}. Options: {list(MODEL_OPTIONS.keys())}"
        logger.error(err)
        return jsonify({'error': err}), 400

    if version not in MODEL_OPTIONS[model]:
        err = f"Unexpected version: {version} for model={model}. Options: {list(MODEL_OPTIONS[model].keys())}"
        logger.error(err)
        return jsonify({'error': err}), 400
    
    # Metric3D specific logic
    if model == METRIC_3D:
        if focal_length is None:
            return jsonify({'error': 'Missing required parameter: focal_length'}), 400
        try:
            focal_length = float(focal_length)
        except ValueError:
            return jsonify({'error': 'Invalid focal_length. Must be a number.'}), 400

    # Marigold specific logic
    if model == MARIGOLD:
        if denoising_steps is None:
            logger.info(f"'denoising_steps' parameter not received, reverting to default={DEFAULT_DENOISING_STEPS}")
            denoising_steps = DEFAULT_DENOISING_STEPS
        if ensemble_size is None:
            ensemble_size = DEFAULT_ENSEMBLE_SIZE
            logger.info(f"'ensemble_size' parameter not received, reverting to default={DEFAULT_ENSEMBLE_SIZE}")
            
    # Decode image
    try:
        img = decode_image(request)
    except Exception as e:
        logger.exception("Failed to decode image")
        return jsonify({'error': f'Failed to decode image: {e}'}), 400
    
    # Run inference
    try:
       depth_map, confidence = model_inference(img,
                                                 version,
                                                 model,
                                                 focal_length,
                                                 denoising_steps,
                                                 ensemble_size)

    except Exception as e:
        logger.exception("Inference failed")
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500

    # Encode result to .npz
    try:
        buffer = io.BytesIO()
        if depth_map is not None and confidence is not None:
            np.savez(buffer, depth_map=depth_map, confidence=confidence)
        else:
            np.savez(buffer, depth_map=depth_map)
        
        buffer.seek(0)
        
    except Exception as e:
        logger.exception("Failed to serialize inference results")
        return jsonify({'error': 'Failed to serialize inference results'}), 500

    # Return file
    return send_file(
        buffer,
        as_attachment=True,
        download_name='depth_and_confidence.npz',
        mimetype='application/octet-stream'
    )

def decode_image(request) -> ImageNP:
    image_bytes = request.files['image'].read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img: ImageNP = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Invalid image format")

    return img

def model_inference(img : ImageNP,
                  version : str,
                  model : str,
                  focal_length : float,
                  denoising_steps : int,
                  ensemble_size : int) -> Tuple[np.ndarray, np.ndarray]:
    
    depth_map, confidence = None, None

    if model == METRIC_3D:
        depth_map, confidence = Metric3D_inference.monocular_depth_estimation(
            version=version,
            org_rgb=img,
            focal_length_px=focal_length
        )

    elif model == MARIGOLD:
        depth_map = Marigold_inference.monocular_depth_estimation(img, 
                                                                  denoise_steps=denoising_steps,
                                                                  ensemble_size=ensemble_size)
        confidence = None  # Marigold does not return confidence
    
    if model == MIDAS:
        depth_map = MiDaS_inference.monocular_depth_estimation(img)
        confidence = None  # MiDaS does not return confidence
        
    assert depth_map.size[0:2] == img.size[0:2]
    assert confidence is None or confidence.size == img.size[0:2]
    
    return depth_map, confidence
