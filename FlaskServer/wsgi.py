import os
import sys
from flask import Flask, request, jsonify, send_file
import io
import logging
import numpy as np
import cv2

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./Metric3D"))
sys.path.append(os.path.abspath("./Marigold"))
sys.path.append(os.path.abspath("./FlaskServer"))

from FlaskServer import marigold_inference
import metric3d_inference
from metric3d_inference import Image

# --------------------- Logger Setup ---------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --------------------- Flask Setup ---------------------
app = Flask(__name__)

# --------------------- Model Config ---------------------
METRIC_3D = 'metric3d'
MARIGOLD = 'marigold'

MODEL_OPTIONS = {
    METRIC_3D: {
        'small': 'metric3d_vit_small',
        'large': 'metric3d_vit_large',
        'giant': 'metric3d_vit_giant2',
    },
    MARIGOLD: {
        'default': 'default'
    }
}

# --------------------- Route ---------------------
@app.route("/inference/", methods=['POST'])
def run_inference():
    """
    Run MDE inference on an uploaded image.
    Run with:
        flask --app wsgi run --host=0.0.0.0
        gunicorn --bind 0.0.0.0:5000 wsgi:app
    """

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    json_data = request.form.to_dict()
    model = json_data.get('model')
    version = json_data.get('version')
    focal_length = json_data.get('focal_length')

    # Validate model & version
    if model not in MODEL_OPTIONS:
        err = f"Unexpected model: {model}. Options: {list(MODEL_OPTIONS.keys())}"
        logger.error(err)
        return jsonify({'error': err}), 400

    if version not in MODEL_OPTIONS[model]:
        err = f"Unexpected version: {version} for model={model}. Options: {list(MODEL_OPTIONS[model].keys())}"
        logger.error(err)
        return jsonify({'error': err}), 400

    if model == METRIC_3D:
        if focal_length is None:
            return jsonify({'error': 'Missing required parameter: focal_length'}), 400
        try:
            focal_length = float(focal_length)
        except ValueError:
            return jsonify({'error': 'Invalid focal_length. Must be a number.'}), 400

    # Log request parameters
    logger.info("Received request with parameters:")
    for key, value in json_data.items():
        logger.info(f"  {key}: {value}")

    # Decode image
    try:
        image_bytes = request.files['image'].read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        img: Image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Invalid image format")

    except Exception as e:
        logger.exception("Failed to decode image")
        return jsonify({'error': f'Failed to decode image: {e}'}), 400

    # Run inference
    try:
        depth_map, confidence = None, None

        if model == METRIC_3D:
            depth_map, confidence = metric3d_inference.estimate_depth(
                version=version,
                org_rgb=img,
                focal_length_px=focal_length
            )

        elif model == MARIGOLD:
            depth_map = marigold_inference.estimate_depth(img)
            confidence = None  # Marigold does not return confidence

    except Exception as e:
        logger.exception("Inference failed")
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500

    # Encode result to .npz
    try:
        buffer = io.BytesIO()
        np.savez(buffer, depth_map=depth_map, confidence=confidence)
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
