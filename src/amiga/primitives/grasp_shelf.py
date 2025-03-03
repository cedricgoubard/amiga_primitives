import time
import random
from os.path import join

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
        grasp_module: GraspingLightningModule = None,
        detect_obj_from_overlook: bool = True,
        fall_back_after_grasp: bool = True,
        already_grasped: list = [],
        img_save_path: str = ""
        ):

    if obj_name is not None and len(already_grasped) > 0:
        assert obj_name not in already_grasped, f"{obj_name} already grasped"
    
    ####################################################################################
    ############# Find the object, get an position estimate and move closer ############
    ####################################################################################
    if detect_obj_from_overlook:
        
        ########################## Move to overlook position ###########################
        q_target = robot.get_named_joints_cfg(name="overlook")[:6]
        q_now = robot.get_observation()["joint_positions"][:6]
        # Check if already in overlook position
        if np.allclose(q_target, q_now, atol=0.1):
            print("Already in overlook position")
        else:
            res = robot.go_to_joint_positions_through_safe_point(
                joint_positions=q_target, wait=True
                )  
            if res != True: raise ValueError("Failed to move to overlook position")

        robot.open_gripper()
     
        ################################ Detect object #################################
        rgb, depth = camera.read()
        objs = detector(rgb, tracking=False)

        # For debug
        save_rgb(rgb, path=join(img_save_path, "latest_rgb.jpg"))
        save_depth(depth, path=join(img_save_path, "latest_depth.jpg"), max=3000)
        if len(objs) > 0: 
            save_rgb(overlay_results(rgb, objs), path=join(img_save_path, "latest_objects.jpg"))

        if obj_name is None:
            available_objs = [oo for oo in objs if oo["class_name"] not in already_grasped]
            if len(available_objs) == 0: raise ValueError("No objects to grasp")
            det_obj = [random.choice(available_objs)] 
        else:
            det_obj = [obj for obj in objs if obj["class_name"] == obj_name]
            if len(det_obj) == 0: raise ValueError(f"No {obj_name} detected")

        print(f"Grasping {det_obj[0]['class_name']}")


        ############################## Compute 3D coords ###############################
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


        ####################### Get a closer look of the object ########################
        target = obj_bl_coords
        target[1] += 0.43  # offset in y (backwards)
        target[2] += 0.065  # offset in z (upwards)

        if target[2] < 0.4:
            wp = robot.get_named_eef_position(name="low_wp")
        else:
            wp = robot.get_named_eef_position(name="high_wp")
        

        robot.follow_eef_position_path_default_orientation(path=[wp, target], wait=True, blend=[0.1, 0.0])
        # time.sleep(0.5)


    ####################################################################################
    ###################################### Grasp #######################################
    ####################################################################################
    init_xyz = robot.get_observation()["ee_pose_euler"][:3]
    rgb_pre_grasp, depth_pre_grasp = camera.read()
    
    if grasp_module is None:
        # TODO: use obj det to refine position, and grasp
        raise NotImplementedError("Grasping module not provided")

    rgb = cv2.resize(
        rgb_pre_grasp, (grasp_module.cfg.img_size, grasp_module.cfg.img_size))
    depth = cv2.resize(
        depth_pre_grasp, (grasp_module.cfg.img_size, grasp_module.cfg.img_size))

    rgb = (rgb / 255.0)
    depth = np.clip(depth, 0, grasp_module.cfg.max_depth_mm) / grasp_module.cfg.max_depth_mm

    res = grasp_module.pred_dx_dy_dz(rgb, depth)
    target_xyz = init_xyz + res
    target_xyz[1] += 0.09  # offset in y (backwards)
    target_xyz[2] += 0.08  # offset in z (upwards)

    robot.go_to_eef_position_default_orientation(eef_position=target_xyz, wait=True)
    robot.close_gripper()
    time.sleep(0.8)
    

    ####################################################################################
    #################################### Fall back #####################################
    ####################################################################################
    if fall_back_after_grasp:
        fall_back_xyz = target_xyz.copy()
        fall_back_xyz[1] += 0.3
        fall_back_xyz[2] += 0.07

        eef_pose = robot.get_observation()["ee_pose_euler"]
        fall_back_pose = np.concatenate([fall_back_xyz, eef_pose[3:]])
        fall_back_q = robot.get_ik(eef_pose=fall_back_pose)
        
        overlook_q = robot.get_named_joints_cfg(name="overlook")[:6]
        path = np.stack([fall_back_q, overlook_q])
        robot.follow_joint_positions_path(path=path, wait=True, blend=[0.1, 0.0])

    #################################### Place back ####################################
    # target_xyz[1] -= 0.3
    # target_xyz[2] -= 0.07

    # robot.go_to_eef_position_default_orientation(eef_position=target_xyz, wait=True)
    # robot.open_gripper()
    # time.sleep(1.0)
    # robot.go_to_eef_position_default_orientation(eef_position=fall_back_xyz, wait=True)

    # key = None
    # while key != "c":
    #     key = input("Press 'c' to continue: ")
    #     key = key.lower()

    return det_obj[0]["class_name"]
