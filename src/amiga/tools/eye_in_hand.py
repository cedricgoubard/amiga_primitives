import json
from datetime import datetime
import time
import argparse

import cv2
from omegaconf import OmegaConf
import numpy as np

from amiga.drivers.cameras import ZEDCamera, RealsenseCamera  # DO NOT REMOVE
from amiga.drivers.amiga import AMIGA


# Define global variables
samples = []  # To store (charuco_pose, eef_pose) pairs

# Helper function to detect Charuco board and estimate pose
def get_charuco_pose(image, charuco_dict, charuco_board, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, charuco_dict)
    
    if ids is not None:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
        if charuco_ids is not None and len(charuco_ids) > 4:  # Require enough points for pose estimation
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs
            )
            if success:
                return np.hstack([rvec, tvec])  # [rotation vector | translation vector]
    return None


# Function to save samples
def save_samples(filename="hand_eye_samples.json"):
    with open(filename, 'w') as f:
        json.dump(samples, f, indent=4)
    print(f"Samples saved to {filename}")


def load_samples(filename="hand_eye_samples.json"):

# Hand-eye calibration optimization
def perform_calibration():
    if len(samples) < 3:  # At least 3 samples are needed for calibration
        print("Not enough samples. Collect at least 3.")
        return
    
    obj_poses = []
    eef_poses = []
    
    for sample in samples:
        obj_poses.append(sample['charuco_pose'])
        eef_poses.append(sample['eef_pose'])
    
    obj_poses = np.array(obj_poses, dtype=np.float64)
    eef_poses = np.array(eef_poses, dtype=np.float64)
    
    r_cam2obj, t_cam2obj = obj_poses[:, :3], obj_poses[:, 3:]
    r_eef2base, t_eef2base = eef_poses[:, :3], eef_poses[:, 3:]
    
    r_cam2eef, t_cam2eef = cv2.calibrateHandEye(
        r_eef2base, t_eef2base, r_cam2obj, t_cam2obj, method=cv2.CALIB_HAND_EYE_TSAI
    )
    print("Calibration complete!")
    print("Rotation matrix (camera to end-effector):\n", r_cam2eef)
    print("Translation vector (camera to end-effector):\n", t_cam2eef)
    return r_cam2eef, t_cam2eef


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file", required=True)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    # camera client
    backend = eval(cfg.cam_zmq.class_name)(cfg)
    camera: ZEDCamera = backend.make_zmq_client(cfg.cam_zmq.port, cfg.cam_zmq.host, async_method="read")

    # Make robot client
    rob_backend = eval(cfg.robot_zmq.class_name)(cfg.robot_zmq)
    robot: AMIGA = rob_backend.make_zmq_client(cfg.robot_zmq.port, cfg.robot_zmq.host)
    robot.set_freedrive_mode(enable=True)
    robot.close_gripper()

    imgs = None
    print("Waiting for image...")   
    while imgs is None:
        imgs = camera.read()
        if isinstance(imgs, dict) and "error" in imgs.keys():
            print("Error: ", imgs)
            imgs = None
            time.sleep(5)
        time.sleep(0.1)
    print(f"Got image; moving on...")
    
    params = camera.get_parameters()
    # Camera intrinsic parameters (assume pre-calibrated)
    camera_matrix = np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]])  # Replace with actual values
    dist_coeffs = np.zeros((5,))  # Replace with actual distortion coefficients if available

    # Charuco board parameters
    squares_x = 5  # Number of squares in the x direction
    squares_y = 7  # Number of squares in the y direction
    square_length = 0.027  # Square side length in meters
    marker_length = 0.02  # Marker side length in meters

    # Use getPredefinedDictionary for compatibility with newer OpenCV versions
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    charuco_board = cv2.aruco.CharucoBoard((squares_y, squares_x), square_length, marker_length, charuco_dict)

    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.cameraMatrix = camera_matrix
    charuco_params.distCoeffs = dist_coeffs

    detector_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.CharucoDetector(board=charuco_board, charucoParams=charuco_params, detectorParams=detector_params)

    try:
        while True:
            # Capture image
            img, _ = camera.read()  # Function to get an image from the camera
            cv2.imwrite(f"latest_rgb.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

            # print(charuco_corners, charuco_ids) 
           
            if charuco_corners is not None and len(charuco_ids) >= 4:
  
                img_with_markers = cv2.aruco.drawDetectedMarkers(img.copy(), marker_corners, marker_ids)
                img_with_charuco = cv2.aruco.drawDetectedCornersCharuco(img_with_markers, charuco_corners, charuco_ids, (0, 255, 0))
                cv2.imwrite(f"latest_charuco.jpg", cv2.cvtColor(img_with_charuco, cv2.COLOR_BGR2RGB))

                eef_pose = robot.get_observation()["ee_pose_euler"]

                # Wait for user input to save the sample
                key = input("Sample collected; what next? Save / Discard / Freedrive / Lock / Quit: ")

                if key.lower() == 's':
                    board_pose = None
                    obj_pts, img_pts = charuco_board.matchImagePoints(charuco_corners, charuco_ids)
                    sth, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
                    board_pose = np.hstack([rvec, tvec])

                    # Save the sample
                    sample = {
                        "timestamp": datetime.now().isoformat(),
                        "charuco_pose": board_pose.tolist(),
                        "eef_pose": eef_pose.tolist(),
                    }
                    samples.append(sample)
                    save_samples()
                    print(f"Sample collected at {sample['timestamp']}")
                
                elif key.lower() == 'f':
                    robot.freedrive()
                
                elif key.lower() == 'l':
                    robot.lock()

                elif key.lower() == 'd':
                    continue

                else:
                    break
            else:
                time.sleep(0.3)
                

    except KeyboardInterrupt:
        print("Terminating...")
    finally:
        cv2.destroyAllWindows()
        if len(samples) > 0:
            perform_calibration()

