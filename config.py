# config.py
import os

# Configuration variables
CALIBRATION_FOLDER = "calibration_data"
OUTPUT_FOLDER = "output"
CAMERA_IDS = [0,1] # Initial default
CAMERA_COUNT = 2

# Calibration Board Parameters
BOARD_ROWS = 6
BOARD_COLUMNS = 9
CHECKER_WIDTH_MM = 25  # Checker width in millimeters
BOARD_SIZE = (BOARD_COLUMNS, BOARD_ROWS) #Chess board size

# Flask ports
CALIBRATION_PORT = 5001
POSE_2D_PORT = 5002
POSE_3D_PORT = 5003
MAIN_PORT = 5000

# Create folders if they don't exist
os.makedirs(CALIBRATION_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Calibration Parameters (Placeholder, will be updated after calibration)
INTRINSIC_PARAMS = {}
EXTRINSIC_PARAMS = {}
CALIBRATED = False