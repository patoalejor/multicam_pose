# utils.py
import cv2
import numpy as np
import os
from config import CALIBRATION_FOLDER, INTRINSIC_PARAMS, EXTRINSIC_PARAMS, CALIBRATED
import pickle

def get_camera_feed(camera_id):
    """Capture a frame from the specified camera.
    
    Args:
        camera_id (int): The camera index.

    Returns:
        tuple: A tuple containing (success_flag, frame).
               If success_flag is False, frame is None.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return False, None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False, None
    return True, frame

def collect_calibration_images(camera_ids, num_images=20, image_delay=1):
    """Collects images for calibration from multiple cameras."""
    
    all_images = {}
    for cam_id in camera_ids:
        all_images[cam_id] = []

    image_count = 0
    
    while image_count < num_images:
        for cam_id in camera_ids:
            success, frame = get_camera_feed(cam_id)
            if success:
                cv2.imshow(f"Camera {cam_id}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):  # Press 'c' to capture image
                    
                    filename = os.path.join(CALIBRATION_FOLDER, f"cam_{cam_id}_img_{image_count}.jpg")
                    cv2.imwrite(filename, frame)
                    all_images[cam_id].append(filename)
                    image_count+=1
                    print(f"Captured image {image_count}/{num_images} from Camera {cam_id}")
                    
                if key == ord('q'):  # Press 'q' to quit
                    return None, None
            else:
                print(f"Error reading camera: {cam_id}")
                return None, None

    cv2.destroyAllWindows()
    return all_images, image_count

def calibrate_cameras(all_images):
    """Calibrates cameras using collected images and saves calibration parameters."""
    
    if not all_images or not all_images[list(all_images.keys())[0]]:
        print("No images collected for calibration.")
        return False, {} , {}

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    board_size = (9,6) # Chess board size
    
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = {}  # 2d points in image plane.
    
    for cam_id, image_list in all_images.items():
        imgpoints[cam_id] = []
        for filename in image_list:
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints[cam_id].append(corners2)

    if not objpoints:
        print("No corners found in any image")
        return False, {}, {}
    
    ret_cal, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints[list(imgpoints.keys())[0]], gray.shape[::-1], None, None)
    
    if not ret_cal:
        print("Camera Calibration failed")
        return False, {}, {}
    
    # Save intrinsic parameters
    intrinsic_params = {}
    for cam_id in all_images:
        ret_cam, mtx_cam, dist_cam, rvecs_cam, tvecs_cam = cv2.calibrateCamera(objpoints, imgpoints[cam_id], gray.shape[::-1], None, None)
        intrinsic_params[cam_id] = {'camera_matrix': mtx_cam, 'distortion_coeffs': dist_cam}
    
    # Compute extrinsic parameters (first camera is the reference)
    extrinsic_params = {}
    cam_id_first = list(imgpoints.keys())[0]
    for cam_id in all_images:
         if cam_id == cam_id_first:
             extrinsic_params[cam_id] = {'rotation_vector': np.zeros((3,1)), 'translation_vector': np.zeros((3,1))}
         else:    
            _, rvec, tvec = cv2.solvePnP(objp, imgpoints[cam_id][0], mtx_cam, dist_cam)
            extrinsic_params[cam_id] = {'rotation_vector': rvec, 'translation_vector': tvec}

    print("Cameras calibrated successfully.")
    return True, intrinsic_params, extrinsic_params

def save_calibration_data(intrinsic_params, extrinsic_params):
    """Saves calibration data to disk."""
    
    calibration_file = os.path.join(CALIBRATION_FOLDER, "calibration_data.pkl")
    calibration_data = {
        'intrinsic_params': intrinsic_params,
        'extrinsic_params': extrinsic_params
    }
    
    with open(calibration_file, 'wb') as f:
        pickle.dump(calibration_data, f)
        
def load_calibration_data():
    """Loads calibration data from disk."""
    calibration_file = os.path.join(CALIBRATION_FOLDER, "calibration_data.pkl")
    if not os.path.exists(calibration_file):
        print("Calibration file not found.")
        return None, None
    try:
         with open(calibration_file, 'rb') as f:
            calibration_data = pickle.load(f)
         return calibration_data['intrinsic_params'], calibration_data['extrinsic_params']
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None, None