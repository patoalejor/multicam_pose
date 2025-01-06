# calibration.py
from flask import Flask, request, jsonify
import time
import cv2
import threading
import numpy as np
from utils import get_camera_feed, collect_calibration_images, calibrate_cameras, save_calibration_data, load_calibration_data
from config import CALIBRATION_PORT, CAMERA_IDS, CALIBRATION_FOLDER, INTRINSIC_PARAMS, EXTRINSIC_PARAMS, CALIBRATED
import os

app = Flask(__name__)

calibration_lock = threading.Lock()
calibration_process_active = False
CALIBRATED = False

def process_calibration(camera_ids):
    """Handles the calibration process."""
    global CALIBRATED, INTRINSIC_PARAMS, EXTRINSIC_PARAMS
    
    print("Starting calibration process")
    all_images, num_images = collect_calibration_images(camera_ids, num_images=20)
    if all_images and num_images:
        success, intrinsic_params, extrinsic_params = calibrate_cameras(all_images)
        if success:
            save_calibration_data(intrinsic_params, extrinsic_params)
            INTRINSIC_PARAMS = intrinsic_params
            EXTRINSIC_PARAMS = extrinsic_params
            CALIBRATED = True
            print("Calibration completed.")
            return True
        else:
            print("Calibration failed.")
            return False
    else:
        print("Calibration cancelled or error occurred.")
        return False

def start_calibration(camera_ids):
    """Starts the calibration thread."""
    global calibration_process_active
    
    with calibration_lock:
        if calibration_process_active:
            return False
        calibration_process_active = True

    try:
        calibration_thread = threading.Thread(target=process_calibration, args=(camera_ids,))
        calibration_thread.start()
        return True
    finally:
       with calibration_lock:
        calibration_process_active = False
            
@app.route('/check_calibration', methods=['GET'])
def check_calibration():
    """Check if the cameras are calibrated."""
    
    global CALIBRATED, INTRINSIC_PARAMS, EXTRINSIC_PARAMS
    
    if not CALIBRATED:
        if os.path.exists(os.path.join(CALIBRATION_FOLDER, "calibration_data.pkl")):
          INTRINSIC_PARAMS, EXTRINSIC_PARAMS = load_calibration_data()
          if INTRINSIC_PARAMS and EXTRINSIC_PARAMS:
              CALIBRATED = True
    
    return jsonify({'is_calibrated': CALIBRATED})

@app.route('/start_calibration', methods=['POST'])
def start_calibration_route():
    """Starts the calibration process."""
    
    data = request.get_json()
    if data and 'camera_ids' in data:
        camera_ids = data['camera_ids']
    else:
        return jsonify({'message': 'Camera IDs are not provided'}), 400
    
    if start_calibration(camera_ids):
        return jsonify({'message': 'Calibration started'}), 202
    else:
        return jsonify({'message': 'Calibration in progress or failed to start'}), 400

@app.route('/get_frame', methods=['POST'])
def get_frame():
    """Returns a frame from each specified camera."""
    
    data = request.get_json()
    if not data or 'camera_ids' not in data:
       return jsonify({'error': 'Camera IDs are missing'}), 400
    
    camera_ids = data['camera_ids']
    frames = {}

    for cam_id in camera_ids:
        success, frame = get_camera_feed(cam_id)
        if success:
           _, buffer = cv2.imencode('.jpg', frame)
           frames[cam_id] = buffer.tobytes()
        else:
            frames[cam_id] = None  # Indicate error
    
    return jsonify(frames)

if __name__ == '__main__':
    app.run(debug=False, port=CALIBRATION_PORT)