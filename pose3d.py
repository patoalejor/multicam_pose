# 3dpose.py
from flask import Flask, request, jsonify
import numpy as np
import time
from config import POSE_3D_PORT, INTRINSIC_PARAMS, EXTRINSIC_PARAMS, CALIBRATED
from utils import load_calibration_data
app = Flask(__name__)


def triangulate_points(keypoints_2d, intrinsic_params, extrinsic_params):
  """Triangulate 2D points from multiple views to 3D points."""

  if not keypoints_2d or not all(keypoints_2d.values()):
      return None
  
  if not CALIBRATED:
      print("Cameras need to be calibrated before getting 3D pose")
      return None
  
  points_3d = {}
  keypoint_ids = keypoints_2d[list(keypoints_2d.keys())[0]].keys()

  for keypoint_id in keypoint_ids:
    points_2d_cam = []
    for cam_id, keypoints in keypoints_2d.items():
      if keypoints and keypoint_id in keypoints:
        points_2d_cam.append(np.array(keypoints[keypoint_id]))
      else:
        points_2d_cam = []
        break

    if not points_2d_cam or len(points_2d_cam) < 2:
         points_3d[keypoint_id] = None
         continue
      
    proj_matrices = []
    for cam_id, keypoints in keypoints_2d.items():
       if keypoints and keypoint_id in keypoints:
            camera_matrix = intrinsic_params[cam_id]['camera_matrix']
            rvec = extrinsic_params[cam_id]['rotation_vector']
            tvec = extrinsic_params[cam_id]['translation_vector']
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            proj_mat = np.dot(camera_matrix, np.concatenate((rotation_matrix, tvec), axis=1))
            proj_matrices.append(proj_mat)
    
    if len(proj_matrices) < 2:
       points_3d[keypoint_id] = None
       continue
    
    points_2d_cam_h = [np.append(point, 1) for point in points_2d_cam]
    points_2d_np = np.array(points_2d_cam_h).T # 3 x n matrix where n is the number of cameras
    proj_matrices_np = np.array(proj_matrices) # n x 3 x 4 matrix where n is the number of cameras

    A = np.concatenate([np.dot(proj_matrices_np[i][0:3,0:3].T,proj_matrices_np[i][0:3,:].T)-points_2d_np[i,:].reshape((1,3)).T for i in range(len(proj_matrices_np))])

    _, _, V = np.linalg.svd(A)
    point_3d_homo = V[-1, :] / V[-1, 3]
    points_3d[keypoint_id] = point_3d_homo[0:3] # Return 3d point
    
  return points_3d

@app.route('/get_3d_pose', methods=['POST'])
def get_3d_pose_route():
    """Gets 3D pose keypoints given 2D keypoints and calibration data."""
   
    data = request.get_json()
    if not data or 'keypoints_2d' not in data:
       return jsonify({'error': 'No 2D keypoints received'}), 400

    keypoints_2d = data['keypoints_2d']
    global INTRINSIC_PARAMS, EXTRINSIC_PARAMS, CALIBRATED
    
    if not CALIBRATED:
        INTRINSIC_PARAMS, EXTRINSIC_PARAMS = load_calibration_data()
        CALIBRATED = True
    
    points_3d = triangulate_points(keypoints_2d, INTRINSIC_PARAMS, EXTRINSIC_PARAMS)
    
    return jsonify({'points_3d': points_3d})

if __name__ == '__main__':
    app.run(debug=False, port=POSE_3D_PORT)