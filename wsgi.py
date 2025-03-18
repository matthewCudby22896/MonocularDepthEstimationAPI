from flask import Flask, request, jsonify, send_file
from markupsafe import escape
import io
import metric3d_inference
import numpy as np
import cv2
import logging
from metric3d_inference import Image

# Basic logger definition
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_OPTIONS = {
    'small' : 'metric3d_vit_small',
    'large' : 'metric3d_vit_large',
    'giant' : 'metric3d_vit_giant2',
}

@app.route("/inference/", methods=['POST'])
def run_inference():
    """
    flask --app wsgi run --host=0.0.0.0
    gunicorn --bind 0.0.0.0:5000 wsgi:app
    """
    
    if 'image' not in request.files:
        return jsonify({'error' : 'No image in the request'}), 400
    
    # Extract the request body
    json_data : dict = request.form.to_dict()
    
    # Extract request parameters
    focal_length = json_data.get('focal_length', None)
    version = json_data.get('version', None)
    
    if focal_length is None:
        return jsonify({'error': 'Missing parameter: focal_length'}), 400

    if version is None:
        return jsonify({'error': 'Missing parameter: version'}), 400
    
    if version not in MODEL_OPTIONS:
        return jsonify({'error' : f'Unrecognised version: {version=}, available options={MODEL_OPTIONS.keys()}'}), 400
    
    # Log the request
    body_str : str = ''.join([f"\n\t{key} : {value}" for key, value in json_data.items()])
    logger.info(f"Request Body: {body_str}")

    # Access and decode the sent image into an OpenCV format
    image_bytes : bytes = request.files['image'].read() # byte file
    npimg  : np.ndarray = np.frombuffer(image_bytes, np.uint8) # convert bytes into a numpy array
    img : Image = cv2.imdecode(npimg, cv2.IMREAD_COLOR) # converts into format that opencv can process

    depth_map = metric3d_inference.estimate_depth(version, org_rgb=img, focal_length_px=focal_length)
    
    # Save depth map to a binary buffer
    buffer = io.BytesIO()
    np.save(buffer, depth_map)
    buffer.seek(0)  # Move to the beginning of the buffer

    return send_file(buffer, as_attachment=True, download_name='depth_map.npy', mimetype='application/octet-stream')
