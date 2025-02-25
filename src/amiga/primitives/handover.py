import time
import random
from typing import List

import torch
import numpy as np
import transforms3d as t3d
import datetime as dt
import cv2

from amiga.vision import KitchenObjectDetector, overlay_results
from amiga.drivers.cameras import ZEDCamera , CameraDriver, CameraParameters # DO NOT REMOVE, used at eval
from amiga.drivers.amiga import AMIGA  # DO NOT REMOVE, used at eval
from amiga.vision import overlay_results
from amiga.models import GraspingLightningModule
from amiga.utils import save_rgb, save_depth


def handover(
        camera: CameraDriver, 
        robot: AMIGA,
        position: np.ndarray = [0.0, -1.1, 0.7],
        initial_path: List[np.ndarray] = None,
        initial_blend: List[float] = None
        ):

    if initial_path is None: initial_path = []
    if initial_blend is None: initial_blend = []
    
    path = initial_path
    blend = initial_blend

   
    # Add handover point to initial path
    path.append(np.array(position))
    blend.append(0.0)


    robot.follow_eef_position_path_default_orientation(path=path, wait=True, blend=blend)

    robot.set_freedrive_mode(enable=True)

    rgb, depth = camera.read()
    save_rgb(rgb, "latest_rgb.jpg")
    save_depth(depth, "latest_depth.jpg")

    contact_detected = False
    start_position = robot.get_observation()["ee_pose_euler"][:3]
    while not contact_detected:
        obs = robot.get_observation()

        # If we detect a displacement, we're in contact
        if np.linalg.norm(np.array(obs["ee_pose_euler"][:3]) - np.array(start_position)) > 0.02:
            contact_detected = True

        time.sleep(0.1)

    robot.open_gripper()
    time.sleep(2.0)
    robot.close_gripper()
    # time.sleep(1.0)

    robot.set_freedrive_mode(enable=False)
    