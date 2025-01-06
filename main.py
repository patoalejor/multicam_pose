# main.py
import sys
import requests
import json
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QGridLayout, QWidget,
                            QPushButton, QLineEdit, QHBoxLayout, QVBoxLayout, QFileDialog, QComboBox, QMessageBox, QProgressBar)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import pyqtgraph as pg
from config import MAIN_PORT, CALIBRATION_PORT, POSE_2D_PORT, POSE_3D_PORT, CAMERA_IDS
import os
from ui.mainwindow import Ui_MainWindow

class CameraStreamThread(QThread):
    frame_ready = pyqtSignal(int, QPixmap)

    def __init__(self, camera_ids, parent=None):
        super().__init__(parent)
        self.camera_ids = camera_ids
        self.running = True
        self.frames = {}

    def run(self):
      while self.running:
            try:
                frames = self.get_frames_from_calibration()
                if frames:
                  for cam_id, frame_bytes in frames.items():
                      if frame_bytes:
                        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

                        if frame is not None and frame.size > 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w, ch = frame_rgb.shape
                            img = QImage(frame_rgb.data, w, h, QImage.Format_RGB888)
                            pix = QPixmap.fromImage(img)
                            self.frame_ready.emit(cam_id, pix)
            except Exception as e:
                print(f"Error during camera streaming: {e}")
            time.sleep(0.03) # Keep the frame rate low
    
    def get_frames_from_calibration(self):
        try:
            response = requests.post(f'http://localhost:{CALIBRATION_PORT}/get_frame',
                                     json={'camera_ids': self.camera_ids})
            if response.status_code == 200:
                return response.json()
            else:
              print("Error from calibration server: ", response.status_code)
              return None
        except Exception as e:
            print(f"Error requesting frames from calibration: {e}")
            return None
    
    def stop(self):
       self.running = False

class PoseEstimationThread(QThread):
    pose_ready = pyqtSignal(dict)

    def __init__(self, camera_ids, output_folder, parent=None):
        super().__init__(parent)
        self.camera_ids = camera_ids
        self.output_folder = output_folder
        self.running = True
        self.frames = {}
        self.keypoints_2d = {}
        self.points_3d = {}

    def run(self):
      while self.running:
            try:
               self.process_pose()
               self.pose_ready.emit(self.points_3d)
            except Exception as e:
                print(f"Error during pose estimation: {e}")
            time.sleep(0.05) # Keep the frame rate low
    
    def get_frames(self):
        try:
            response = requests.post(f'http://localhost:{CALIBRATION_PORT}/get_frame',
                                     json={'camera_ids': self.camera_ids})
            if response.status_code == 200:
                return response.json()
            else:
                print("Error from calibration server: ", response.status_code)
                return None
        except Exception as e:
            print(f"Error requesting frames from calibration: {e}")
            return None

    def get_2d_pose(self, frames):
        try:
             response = requests.post(f'http://localhost:{POSE_2D_PORT}/get_2d_pose',
                                       json={'frames': frames})
             if response.status_code == 200:
                return response.json()
             else:
                print("Error from 2dpose server: ", response.status_code)
                return None
        except Exception as e:
            print(f"Error getting 2D pose: {e}")
            return None

    def get_3d_pose(self, keypoints_2d):
       try:
            response = requests.post(f'http://localhost:{POSE_3D_PORT}/get_3d_pose',
                                    json={'keypoints_2d': keypoints_2d})
            if response.status_code == 200:
               return response.json().get('points_3d', {})
            else:
              print(f"Error from 3dpose server: {response.status_code}")
              return None
       except Exception as e:
           print(f"Error getting 3D pose: {e}")
           return None
    
    def process_pose(self):
        
         frames = self.get_frames()
         if not frames:
           return

         self.keypoints_2d = self.get_2d_pose(frames)
         if not self.keypoints_2d:
            return

         self.points_3d = self.get_3d_pose(self.keypoints_2d)
    
    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.camera_stream_thread = None
        self.pose_estimation_thread = None
        self.output_folder = None
        self.camera_ids = None
        self.calibrated = False

        self.ui.folderButton.clicked.connect(self.select_output_folder)
        self.ui.startButton.clicked.connect(self.start_process)
        self.ui.cameraComboBox.currentIndexChanged.connect(self.change_camera_ids)
        
        self.plot_widget = pg.PlotWidget()
        self.ui.plotLayout.addWidget(self.plot_widget)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.ui.folderLineEdit.setText(folder)
            self.output_folder = folder
            
    def change_camera_ids(self):
        text_ids = self.ui.cameraComboBox.currentText()
        try:
            self.camera_ids = [int(id) for id in text_ids.split(',')]
        except ValueError:
             self.camera_ids = None
             QMessageBox.warning(self, "Error", "Invalid camera IDs. Use comma-separated integers (e.g., 0,1).")

    def start_process(self):
      
        if not self.output_folder or not self.camera_ids:
            QMessageBox.warning(self, "Error", "Please select output folder and camera IDs.")
            return
        
        self.ui.startButton.setEnabled(False)
        self.ui.progressBar.setValue(0)

        self.check_calibration()
    
    def check_calibration(self):
        
         try:
           self.ui.progressBar.setValue(10)
           response = requests.get(f'http://localhost:{CALIBRATION_PORT}/check_calibration')
           
           if response.status_code == 200:
                data = response.json()
                if data['is_calibrated']:
                   self.calibrated = True
                   self.ui.progressBar.setValue(20)
                   self.start_stream()
                else:
                   self.calibrate_cameras()
           else:
              print("Error from calibration server: ", response.status_code)
              QMessageBox.critical(self, "Error", "Could not communicate with calibration server")
              self.ui.startButton.setEnabled(True)
              self.ui.progressBar.setValue(0)
         except Exception as e:
            print(f"Error checking calibration: {e}")
            QMessageBox.critical(self, "Error", f"Could not check camera calibration {e}")
            self.ui.startButton.setEnabled(True)
            self.ui.progressBar.setValue(0)
          
    def calibrate_cameras(self):
        try:
            self.ui.progressBar.setValue(30)
            response = requests.post(f'http://localhost:{CALIBRATION_PORT}/start_calibration', json={'camera_ids':self.camera_ids})
            
            if response.status_code == 202:
              QMessageBox.information(self, "Calibration", "Calibration started. Please capture images using 'c' key in the opened window and 'q' to finish.")
              self.ui.progressBar.setValue(50)
              self.wait_calibration_finished()
            else:
              QMessageBox.critical(self, "Error", "Could not start calibration.")
              self.ui.startButton.setEnabled(True)
              self.ui.progressBar.setValue(0)
        except Exception as e:
           print(f"Error starting calibration: {e}")
           QMessageBox.critical(self, "Error", f"Could not start camera calibration {e}")
           self.ui.startButton.setEnabled(True)
           self.ui.progressBar.setValue(0)
           
    def wait_calibration_finished(self):
        """Wait for the calibration process to finish."""
       
        def check_calibration_status():
           try:
            response = requests.get(f'http://localhost:{CALIBRATION_PORT}/check_calibration')
            if response.status_code == 200:
               data = response.json()
               if data['is_calibrated']:
                   self.calibrated = True
                   self.ui.progressBar.setValue(70)
                   self.start_stream()
                   return True
               else:
                   return False
            else:
              print("Error from calibration server: ", response.status_code)
              QMessageBox.critical(self, "Error", "Could not communicate with calibration server")
              self.ui.startButton.setEnabled(True)
              self.ui.progressBar.setValue(0)
              return True
           except Exception as e:
                print(f"Error getting calibration status: {e}")
                QMessageBox.critical(self, "Error", f"Could not check camera calibration {e}")
                self.ui.startButton.setEnabled(True)
                self.ui.progressBar.setValue(0)
                return True

        while not check_calibration_status():
            QApplication.processEvents() # To check for user interaction in other thread
            time.sleep(1) # Check status every 1 second
    
    def start_stream(self):
      """Starts the camera stream and pose estimation."""
      
      self.ui.progressBar.setValue(80)
      self.camera_stream_thread = CameraStreamThread(self.camera_ids)
      self.camera_stream_thread.frame_ready.connect(self.update_frames)
      self.camera_stream_thread.start()

      self.pose_estimation_thread = PoseEstimationThread(self.camera_ids, self.output_folder)
      self.pose_estimation_thread.pose_ready.connect(self.update_plot)
      self.pose_estimation_thread.start()
      self.ui.progressBar.setValue(100)
    
    def update_frames(self, cam_id, pixmap):
       if cam_id == 0:
          self.ui.cameraLabel1.setPixmap(pixmap)
       elif cam_id == 1:
          self.ui.cameraLabel2.setPixmap(pixmap)
       
    def update_plot(self, points_3d):
        if points_3d:
          x_values = [point[0] if point is not None else 0 for point in points_3d.values()]
          y_values = [point[1] if point is not None else 0 for point in points_3d.values()]
          z_values = [point[2] if point is not None else 0 for point in points_3d.values()]

          self.plot_widget.clear()
          bar_item = pg.BarGraphItem(x=list(range(len(x_values))), height=z_values, width=0.8, brush='b')
          self.plot_widget.addItem(bar_item)
    
    def closeEvent(self, event):
       
        if self.camera_stream_thread:
           self.camera_stream_thread.stop()
           self.camera_stream_thread.wait()
        if self.pose_estimation_thread:
           self.pose_estimation_thread.stop()
           self.pose_estimation_thread.wait()

        event.accept()

def start_servers():
    """Starts all Flask servers in separate threads."""
    import subprocess
    
    calibration_server = subprocess.Popen(["python", "calibration.py"])
    pose_2d_server = subprocess.Popen(["python", "2dpose.py"])
    pose_3d_server = subprocess.Popen(["python", "3dpose.py"])
    
    return calibration_server, pose_2d_server, pose_3d_server

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    calibration_server, pose_2d_server, pose_3d_server = start_servers()
    time.sleep(2) # Wait for the servers to start
    main()
    
    calibration_server.terminate()
    pose_2d_server.terminate()
    pose_3d_server.terminate()