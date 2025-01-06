# Multi-Camera 3D Pose Estimation System

This project implements a multi-camera system for 3D human pose estimation using OpenCV, MediaPipe, Flask, and PyQt. It captures video feeds from multiple webcams, calibrates the cameras, detects 2D keypoints, and triangulates them to estimate 3D keypoints. The application provides a user interface for selecting camera IDs and an output folder, as well as displaying the video streams and a 3D pose plot.

## Table of Contents
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Calibration Process](#calibration-process)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## Project Structure

The project consists of the following files and folders:

-   `config.py`: Configuration parameters like folder paths, port numbers and camera ids.
-   `utils.py`: Utility functions for camera feed access, calibration, etc.
-   `calibration.py`: Flask server to handle camera calibration and frame access.
-   `2dpose.py`: Flask server for detecting 2D pose keypoints from video frames.
-   `3dpose.py`: Flask server for calculating 3D keypoints from 2D keypoints.
-   `main.py`: Main application and PyQt UI logic.
-   `requirements.txt`: Lists all project dependencies for pip.
-   `ui/`: Contains files for user interface.
    - `mainwindow.py`: Python file generate from Qt designer for UI
    - `resources.py`: Resources for the user interface if any

## Dependencies

The following Python libraries are required to run this project. They are listed in the `requirements.txt` file for easy installation.

-   opencv-python
-   flask
-   numpy
-   PyQt5
-   pyqtgraph
-   requests
-   mediapipe

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd multicam_pose
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Main Application:**

    ```bash
    python main.py
    ```

2.  **The PyQt application will launch**
   -   Select an output folder using `Select Folder`.
   -   Write the camera ids you are going to use.
   -   Click the `Start` button.
   -   If the system is not calibrated, a message will be displayed for the user to collect the images by pressing `c` when the chessboard is visible, then `q` to finish the calibration.
   -   The application will show the video stream from the selected cameras and also the bar plot with the results of the 3D pose estimation.

## Configuration

- **`config.py`**

    -   `CALIBRATION_FOLDER`: Specifies the folder where calibration images and data are saved (default: "calibration\_data").
    -   `OUTPUT_FOLDER`: Specifies the folder where outputs are saved (default: "output").
    -   `CAMERA_IDS`: Initial list of camera ids to be used (default: `[0, 1]`).
    -   `CALIBRATION_PORT`: Port number for the calibration Flask server (default: 5001).
    -   `POSE_2D_PORT`: Port number for the 2D pose estimation Flask server (default: 5002).
    -   `POSE_3D_PORT`: Port number for the 3D pose estimation Flask server (default: 5003).
    -   `MAIN_PORT`: Port number for the main Flask server (default: 5000)
    -   `INTRINSIC_PARAMS`, `EXTRINSIC_PARAMS`, and `CALIBRATED` are variables that are modified during the calibration process and stored for future executions.

## Calibration Process

1.  When the application starts, it checks if calibration data exists. If not, it prompts the user to initiate the calibration process.
2.  Pressing the `Start` button will check the calibration status.
3. If calibration is required, the system will show a message for the user to collect the images by pressing `c` when the chessboard is visible in all cameras, then `q` to finish the calibration process.
4.  The collected images are used to calculate intrinsic and extrinsic camera parameters, and they are saved to a file.
5. The application will then start processing the camera stream for 2D and 3D keypoints detection.

## Troubleshooting

-   **Camera Not Detected:** Ensure cameras are properly connected and the correct IDs are used. Check if other applications use the same cameras or if the camera drivers are correctly installed.
-   **Calibration Fails:** Ensure the chessboard is properly visible in the captured images, is not blurry and there are at least 10 images per camera. You should try again with more images or better image quality.
-  **Pose Estimation Not Accurate:** The accuracy depends on the camera calibration, the lighting of the scene and the size of the person in the frame. Make sure the chessboard covers a good portion of the image and that there is no reflection.
-   **Connection Errors:** Check that Flask servers are running on the specified ports. Firewalls or other network settings may also cause problems.
-   **UI Issues:** Ensure the required version of PyQt5 is installed.

## Future Enhancements

-   Improve 3D pose accuracy.
-   Add GUI elements for real-time 3D pose visualization.
-   Implement error checking for pose estimation.
-   Implement a better UI to choose the number of cameras.