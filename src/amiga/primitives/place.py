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


def place(
        robot: AMIGA,
        speed: str = "low",
        start_position: np.ndarray = [0.0, -1.1, 0.7],
        initial_path: List[np.ndarray] = None,
        initial_blend: List[float] = None,
        fall_back_after_place: bool = True
        ):
    assert speed in ["low", "high"], f"Speed {speed} not supported. Choose between 'low' and 'high'."

    if initial_path is None: initial_path = []
    if initial_blend is None: initial_blend = []
    
    path = initial_path
    blend = initial_blend

   
    # Add handover point to initial path
    path.append(np.array(start_position))
    blend.append(0.0)

    robot.set_speed(speed=speed)
    print("Going to position")
    print(path)
    robot.follow_eef_position_path_default_orientation(path=path, wait=True, blend=blend)

    print("Moving")
    robot.move_eef_until_contact(
        direction=np.array([0.0, -1.0, -1.0, 0.0, 0.0, 0.0]),
        )
    print("Contact")
    robot.open_gripper()

    if fall_back_after_place:
        print("Falling back")
        fall_back_pose = robot.get_observation()["ee_pose_euler"]
        fall_back_pose[1] += 0.3
        fall_back_pose[2] += 0.07
        fall_back_q = robot.get_ik(eef_pose=fall_back_pose)
        
        overlook_q = robot.get_named_joints_cfg(name="overlook")[:6]
        print(fall_back_q.shape, overlook_q.shape)
        path = np.stack([fall_back_q, overlook_q])
        robot.follow_joint_positions_path(path=path, wait=True, blend=[0.1, 0.0])
        robot.close_gripper()
        print("Done")
    