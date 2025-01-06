# 2dpose.py
from flask import Flask, request, jsonify
import numpy as np
import time
import cv2
import mediapipe as mp
from config import POSE_2D_PORT, CAMERA_IDS

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.route('/get_2d_pose', methods=['POST'])
def get_2d_pose_route():
    """Gets 2D pose keypoints for a frame from each camera."""
    
    data = request.get_json()
    if not data or 'frames' not in data or not data['frames']:
        return jsonify({'error': 'No frames received'}), 400
    
    frames_data = data['frames']
    
    all_keypoints = {}
    
    for cam_id, frame_data in frames_data.items():
        
        if not frame_data:
            all_keypoints[cam_id] = None
            continue
        try:
          frame_np = np.frombuffer(frame_data, dtype=np.uint8)
          frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
          
          if frame is None or frame.size==0:
              all_keypoints[cam_id] = None
              continue

          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = pose.process(frame_rgb)
         
          if results.pose_landmarks:
            keypoints_2d = {}
            for i, landmark in enumerate(results.pose_landmarks.landmark):
               keypoints_2d[i] = (landmark.x, landmark.y)
            all_keypoints[cam_id] = keypoints_2d
          else:
               all_keypoints[cam_id] = None
        except Exception as e:
             print(f"Error processing frame {cam_id}: {e}")
             all_keypoints[cam_id] = None
    
    return jsonify(all_keypoints)
    
if __name__ == '__main__':
    app.run(debug=False, port=POSE_2D_PORT)