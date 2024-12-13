import time
import random

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


def grasp_from_shelf(
        detector: KitchenObjectDetector, 
        camera: CameraDriver, 
        robot: AMIGA,
        obj_name: str = None, 
        grasp_module: GraspingLightningModule = None
        ):
    
    ############################ Move to overlook position #############################
    q = robot.get_named_joints_cfg(name="overlook")
    robot.open_gripper()
    res = robot.go_to_joint_positions_through_safe_point(joint_positions=q, wait=True)  
    if res != True: raise ValueError("Failed to move to overlook position") 

    # time.sleep(0.2)

    ################################## Detect object ###################################
    rgb, depth = camera.read()
    objs = detector(rgb, tracking=False)

    # For debug
    save_rgb(rgb, path="latest_rgb.jpg")
    save_depth(depth, path="latest_depth.jpg", max=3000)
    if len(objs) > 0: 
        save_rgb(overlay_results(rgb, objs), path="latest_objects.jpg")

    if obj_name is None:
        det_obj = [random.choice(objs)] 
    else:
        det_obj = [obj for obj in objs if obj["class_name"] == obj_name]
        if len(det_obj) == 0: raise ValueError(f"No {obj_name} detected")

    print(f"Grasping {det_obj[0]['class_name']}")


    ################################ Compute 3D coords #################################
    x, y, w, h = det_obj[0]["xywh"]
    w, h = 0.9*w, 0.9*h  # make sure we get mostly the object
    
    depth_box = depth[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    obj_depth = np.median(depth_box)

    # Pixel space to 3D position in camera frame; res is z forward, x up, y left
    obj_cam_coords = camera.uvdepth_to_xyz(u=int(x), v=int(y), depth=obj_depth)

    eef_pose = robot.get_observation()["ee_pose_euler"]  # x left, y back, z up
    bl_to_eef_tf = t3d.affines.compose(
        T=np.array([eef_pose[0], eef_pose[1], eef_pose[2]]),
        R=t3d.euler.euler2mat(eef_pose[3], eef_pose[4], eef_pose[5], axes='sxyz'),
        Z=np.array([1, 1, 1])
        )
    
    eef_cam_trans, eef_cam_rot_mx = robot.get_camera_tf()
    eef_to_cam_tf = t3d.affines.compose(
        T=eef_cam_trans,
        R=eef_cam_rot_mx,
        Z=np.array([1, 1, 1])
        )

    bl_to_cam_tf = np.dot(bl_to_eef_tf, eef_to_cam_tf)

    # Convert object's camera coordinates to homogeneous form
    obj_cam_coords_h = np.array([*obj_cam_coords, 1])

    # Transform object's coordinates to base_link frame
    obj_bl_coords_h = np.dot(bl_to_cam_tf, obj_cam_coords_h)
    obj_bl_coords = obj_bl_coords_h[:3]  # Extract x, y, z from homogeneous coordinates


    ######################### Get a closer look of the object ##########################
    target = obj_bl_coords
    offset = random.uniform(0.35, 0.50)
    target[1] += offset  # offset in y (backwards)
    target[2] += 0.06  # offset in z (upwards)

    robot.go_to_eef_position_through_safe_point(eef_position=target, wait=True)

    # time.sleep(0.5)
    
    init_xyz = robot.get_observation()["ee_pose_euler"][:3]
    rgb, depth = camera.read()


    ############################## Adjust grasp and go in ##############################
    if grasp_module is None:
        # TODO: use obj det to refine position, and grasp
        raise NotImplementedError("Grasping module not provided")

    rgb = cv2.resize(rgb, (grasp_module.cfg.img_size, grasp_module.cfg.img_size))
    depth = cv2.resize(depth, (grasp_module.cfg.img_size, grasp_module.cfg.img_size))

    rgb = (rgb / 255.0)
    depth = np.clip(depth, 0, grasp_module.cfg.max_depth_mm) / grasp_module.cfg.max_depth_mm

    res = grasp_module.pred_dx_dy_dz(rgb, depth)
    target_xyz = init_xyz + res
    print(res[2])
    target_xyz[2] += 0.03  # offset in z (upwards)

    robot.go_to_eef_position_default_orientation(eef_position=target_xyz, wait=True)
    
    
    ############################### Grasp and fall back ################################
    robot.close_gripper()
    time.sleep(1.2)
    fall_back_xyz = init_xyz
    fall_back_xyz[1] += 0.1
    fall_back_xyz[2] = target_xyz[2] + 0.05

    robot.go_to_eef_position_default_orientation(eef_position=fall_back_xyz, wait=True)

    #################################### Place back ####################################
    target_xyz[2] += 0.01

    robot.go_to_eef_position_default_orientation(eef_position=target_xyz, wait=True)
    robot.open_gripper()
    time.sleep(1.0)
    robot.go_to_eef_position_default_orientation(eef_position=fall_back_xyz, wait=True)

    key = None
    while key != "c":
        key = input("Press 'c' to continue: ")
        key = key.lower()
