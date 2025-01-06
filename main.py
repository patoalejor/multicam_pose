# main.py
import sys
import requests
import json
import time
from flask import Flask, render_template, request, jsonify
from config import MAIN_PORT, CALIBRATION_PORT, POSE_2D_PORT, POSE_3D_PORT, CAMERA_IDS
import os
import subprocess
import numpy as np
import cv2
import base64

app = Flask(__name__)

def start_servers():
    """Starts all Flask servers in separate threads."""
    
    # Get the path to the Python interpreter from the virtual environment
    if sys.platform == "win32":
        python_path = os.path.join(sys.prefix, "Scripts", "python.exe")
    else:  
        # Linux or macOS
        python_path = os.path.join(sys.prefix, "bin", "python")

    calibration_server = subprocess.Popen([python_path, "calibration.py"])
    pose_2d_server = subprocess.Popen([python_path, "pose2d.py"])
    pose_3d_server = subprocess.Popen([python_path, "pose3d.py"])
    
    return calibration_server, pose_2d_server, pose_3d_server

@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
       output_folder = request.form.get('output_folder')
       camera_ids_str = request.form.get('camera_ids')
       
       try:
          camera_ids = [int(id) for id in camera_ids_str.split(',')]
       except ValueError:
            return render_template('index.html', error="Invalid camera IDs. Use comma-separated integers (e.g., 0,1).")

       if not output_folder or not camera_ids:
            return render_template('index.html', error="Please select output folder and camera IDs.")
       
       #Check calibration
       is_calibrated = False
       try:
         response = requests.get(f'http://localhost:{CALIBRATION_PORT}/check_calibration')
         if response.status_code == 200:
              data = response.json()
              is_calibrated = data['is_calibrated']
         else:
              return render_template('index.html', error="Could not communicate with calibration server.")
       except Exception as e:
           print(f"Error checking calibration: {e}")
           return render_template('index.html', error=f"Could not check camera calibration {e}")

       if not is_calibrated:
        try:
            response = requests.post(f'http://localhost:{CALIBRATION_PORT}/start_calibration', json={'camera_ids':camera_ids})
            if response.status_code == 202:
               return render_template('index.html', message="Calibration started, please use the opened windows to collect the images")
            else:
               return render_template('index.html', error="Could not start calibration")
        except Exception as e:
            print(f"Error starting calibration: {e}")
            return render_template('index.html', error=f"Could not start camera calibration {e}")

       #Process and stream frames
       try:
           frames = get_frames_from_calibration(camera_ids)
       except Exception as e:
            print(f"Error getting frames: {e}")
            return render_template('index.html', error=f"Could not get the frames {e}")

       if not frames:
             return render_template('index.html', error="No frames were received from calibration module.")
       
       keypoints_2d = get_2d_pose(frames)
       if not keypoints_2d:
            return render_template('index.html', error="No 2D keypoints were received from 2D pose module.")
       
       points_3d = get_3d_pose(keypoints_2d)
       if not points_3d:
             return render_template('index.html', error="No 3D points were received from 3D pose module.")

       #Prepare images to show in the UI
       images = []
       for cam_id, frame_bytes in frames.items():
           if frame_bytes:
             frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
             frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
             if frame is not None and frame.size > 0:
                 _, buffer = cv2.imencode('.jpg', frame)
                 image_base64 = base64.b64encode(buffer).decode('utf-8')
                 images.append(f'data:image/jpeg;base64,{image_base64}')
           else:
              images.append(None)

       x_values = [point[0] if point is not None else 0 for point in points_3d.values()]
       y_values = [point[1] if point is not None else 0 for point in points_3d.values()]
       z_values = [point[2] if point is not None else 0 for point in points_3d.values()]
       
       return render_template('index.html', images=images, x_values=x_values, y_values=y_values, z_values=z_values)
   else:
     return render_template('index.html')
    
def get_frames_from_calibration(camera_ids):
    try:
        response = requests.post(f'http://localhost:{CALIBRATION_PORT}/get_frame',json={'camera_ids': camera_ids})
        if response.status_code == 200:
            return response.json()
        else:
          print("Error from calibration server: ", response.status_code)
          return None
    except Exception as e:
        print(f"Error requesting frames from calibration: {e}")
        return None

def get_2d_pose(frames):
    try:
         response = requests.post(f'http://localhost:{POSE_2D_PORT}/get_2d_pose',json={'frames': frames})
         if response.status_code == 200:
            return response.json()
         else:
            print("Error from 2dpose server: ", response.status_code)
            return None
    except Exception as e:
        print(f"Error getting 2D pose: {e}")
        return None

def get_3d_pose(keypoints_2d):
   try:
        response = requests.post(f'http://localhost:{POSE_3D_PORT}/get_3d_pose', json={'keypoints_2d': keypoints_2d})
        if response.status_code == 200:
           return response.json().get('points_3d', {})
        else:
          print(f"Error from 3dpose server: {response.status_code}")
          return None
   except Exception as e:
       print(f"Error getting 3D pose: {e}")
       return None


if __name__ == '__main__':
    calibration_server, pose_2d_server, pose_3d_server = start_servers()
    time.sleep(2) # Wait for the servers to start
    app.run(debug=False, port=MAIN_PORT)
    
    calibration_server.terminate()
    pose_2d_server.terminate()
    pose_3d_server.terminate()