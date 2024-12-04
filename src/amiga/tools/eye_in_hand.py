import json
from datetime import datetime
import time
import argparse

import cv2
from omegaconf import OmegaConf
import numpy as np
import transforms3d as t3d

from amiga.drivers.cameras import ZEDCamera, RealsenseCamera  # DO NOT REMOVE
from amiga.drivers.amiga import AMIGA


# Define global variables
samples = []  # To store (charuco_pose, eef_pose) pairs
last_dist = 0


def cv2rot_to_rpy(rvec):
    # print(f"rvec: {rvec}")
    # # Use opencv to get rotation matrics instead of vector
    rmx, _ = cv2.Rodrigues(rvec.squeeze())
    # print(f"rmx: {rmx}")
    # Then convert to euler angles
    return np.array(t3d.euler.mat2euler(rmx, axes='sxyz'))
    # print(f"rpy: {rpy}")


def rpy_to_cv2rot(rpy):
    if isinstance(rpy, list):
        rpy = np.array(rpy)

    if rpy.shape == (3,1):
        rpy = rpy.squeeze()

    assert rpy.shape == (3,), f"Invalid shape: {rpy.shape}"
        
    # Convert euler angles to rotation matrix
    rmx = t3d.euler.euler2mat(*rpy, axes='sxyz')
    # Convert rotation matrix to vector
    return cv2.Rodrigues(rmx)[0].squeeze()


# Function to save samples
def save_samples(filename="hand_eye_samples.json"):
    with open(filename, 'w') as f:
        json.dump(samples, f, indent=4)
    # print(f"Samples saved to {filename}")


def load_samples(filename="hand_eye_samples.json"):
    global samples
    try:
        with open(filename, 'r') as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} samples from {filename}")
    except FileNotFoundError:
        print("No saved samples found.")

# Hand-eye calibration optimization
def perform_calibration():
    global samples
    global last_dist
    if len(samples) < 3:  # At least 3 samples are needed for calibration
        print("Not enough samples. Collect at least 3.")
        return
    # print(f"Performing calibration with {len(samples)} samples...")
    
    obj_poses = []
    eef_poses = []
    
    for sample in samples:
        obj_poses.append(sample['charuco_pose'])
        eef_poses.append(sample['eef_pose'])
    
    obj_poses = np.array(obj_poses, dtype=np.float64)
    eef_poses = np.array(eef_poses, dtype=np.float64)
    # print(obj_poses)
    r_cam2obj, t_cam2obj = obj_poses[:, 3:], obj_poses[:, :3]
    r_eef2base, t_eef2base = eef_poses[:, 3:], eef_poses[:, :3]
    
    r_cam2eef, t_cam2eef = cv2.calibrateHandEye(
        r_eef2base, t_eef2base, r_cam2obj, t_cam2obj, method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    if (np.array(r_cam2eef) != np.identity(3)).any():
        r_cam2eef_as_rpy = np.array(t3d.euler.mat2euler(r_cam2eef, axes='sxyz'))
        dist = np.linalg.norm(t_cam2eef)
        print(f"{len(samples)} samples - Last eef-cam dist: {last_dist*100:.1f}cm, current dist: {dist*100:.1f}cm, trans: {[round(x*100, 2) for x in t_cam2eef.squeeze()]}, rot: {[round(x*100, 2) for x in r_cam2eef_as_rpy.squeeze()]}")
        last_dist = dist

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
    dist_coeffs = np.array([params.k1, params.k2, params.p1, params.p2, params.k3])  # Provide zeroes since ZED already rectifies

    # Charuco board parameters
    # You can use https://calib.io/pages/camera-calibration-pattern-generator
    # Rows = x, columns = y, checker width = square_length * 1000
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

    # Load saved samples
    load_samples()

    last_eef = None

    print("Move the end effector around the charuco board to collect samples.")

    try:
        while True:
            # Capture image
            img, _ = camera.read()  # Function to get an image from the camera
            cv2.imwrite(f"latest_rgb.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)
           
            if charuco_corners is not None and len(charuco_ids) >= 4:
  
                img_with_markers = cv2.aruco.drawDetectedMarkers(img.copy(), marker_corners, marker_ids)
                img_with_charuco = cv2.aruco.drawDetectedCornersCharuco(img_with_markers, charuco_corners, charuco_ids, (0, 255, 0))
                cv2.imwrite(f"latest_charuco.jpg", cv2.cvtColor(img_with_charuco, cv2.COLOR_BGR2RGB))

                eef_pose = robot.get_observation()["ee_pose_euler"]
                eef_pose[3:] = rpy_to_cv2rot(eef_pose[3:])
                if last_eef is None:
                    last_eef = eef_pose

                board_pose = None
                obj_pts, img_pts = charuco_board.matchImagePoints(charuco_corners, charuco_ids)
                    
                # if key.lower() == 's':
                if len(obj_pts) >= 24 and np.linalg.norm(eef_pose[:3] - last_eef[:3]) > 0.10:
                    print("Sample collected")
                    # obj_pts is a [ [[y, x, 0]], ..., [[y, x, 0]] ] in charuco coordinate frame (meters) 
                    # with text in normal orientation, origin is top left, x is down, y is right, z is up
                    
                    # img_pts is a [ [[x, y]], ..., [[x, y]] ] in image pixel coordinates

                    sth, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                    board_pose = np.concatenate([tvec.squeeze(), rvec.squeeze()], axis=0).squeeze()

                    # Save the sample
                    sample = {
                        "timestamp": datetime.now().isoformat(),
                        "charuco_pose": board_pose.tolist(),
                        "eef_pose": eef_pose.tolist(),
                    }
                    samples.append(sample)
                    save_samples()

                    if len(samples) > 10:
                        perform_calibration()
                    
                    last_eef = eef_pose
                    time.sleep(0.3)

            else:
                time.sleep(0.3)
                

    except KeyboardInterrupt:
        print("Terminating...")
    finally:
        cv2.destroyAllWindows()
        if len(samples) > 0:
            rot, trans = perform_calibration()
            print("Calibration complete!")
            print("Rotation matrix (camera to end-effector):\n", rot)
            print("Translation vector (camera to end-effector):\n", trans)


