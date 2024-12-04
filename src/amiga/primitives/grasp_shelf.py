import time
import random

import numpy as np
import cv2
import transforms3d as t3d
import datetime as dt

from amiga.vision import KitchenObjectDetector, overlay_results
from amiga.drivers.cameras import ZEDCamera , CameraDriver, CameraParameters # DO NOT REMOVE, used at eval
from amiga.drivers.amiga import AMIGA  # DO NOT REMOVE, used at eval
from amiga.vision import overlay_results


def grasp_from_shelf(obj_name: str, detector: KitchenObjectDetector, camera: CameraDriver, robot: AMIGA):
    q = robot.get_named_joints_cfg(name="overlook")
    res = robot.go_to_joint_positions_through_safe_point(joint_positions=q, wait=True)  
    if res != True: raise ValueError("Failed to move to overlook position") 
    # print(f"Moved to overlook position: {q}")

    key = None
    while key != "c":
        key = input("Press 'c' to continue: ")
        key = key.lower()

    rgb, depth = camera.read()
    objs = detector(rgb)
    cv2.imwrite("latest_rgb.png",cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    # depth = np.clip(depth, 0, 3000)
    # cv2.imwrite("latest_depth.jpg", ((1 - depth / depth.max()) * 255).astype(np.uint8))
    if len(objs) > 0: cv2.imwrite("latest_objects.jpg", overlay_results(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), objs))

    # det_obj = [obj for obj in objs if obj["class_name"] == obj_name]
    # if len(det_obj) == 0: raise ValueError(f"No {obj_name} bottle detected")
    # print(f"Olive oil bottle detected: {olive_oil}")

    det_obj = [random.choice(objs)] 
    print(f"Random object detected: {det_obj[0]['class_name']}")

    x, y, w, h = det_obj[0]["xywh"]
    w, h = 0.9*w, 0.9*h  # make sure we get mostly the object
    # print(f"Object bounding box: {x, y, w, h}")
    # rgb[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)] = [255, 0, 0]  # Draw a red box around the object
    # cv2.imwrite("latest_rgb.jpg",cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    depth_box = depth[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    obj_depth = np.median(depth_box)
    # print(f"Object depth: {obj_depth}")

    # Pixel space to 3D position in camera frame
    obj_cam_coords = camera.uvdepth_to_xyz(u=int(x), v=int(y), depth=obj_depth)
    # print(f"Object coords in camera frame: {obj_cam_coords}")  # z forward, x up, y left

    eef_pose = robot.get_observation()["ee_pose_euler"]  # X Y Z rX rY rZ
    # print(f"EEF pose in base_link: {eef_pose}")  # x left, y back, z up
    bl_to_eef_tf = t3d.affines.compose(
        T=np.array([eef_pose[0], eef_pose[1], eef_pose[2]]),
        R=t3d.euler.euler2mat(eef_pose[3], eef_pose[4], eef_pose[5], axes='sxyz'),
        Z=np.array([1, 1, 1])
        )
    # print(f"TF matrix BL -> EEF: \n{bl_to_eef_tf}")  # x forward, y left, z up
    
    eef_cam_trans, eef_cam_rot_mx = robot.get_camera_tf()
    eef_to_cam_tf = t3d.affines.compose(
        T=eef_cam_trans,
        R=eef_cam_rot_mx,
        Z=np.array([1, 1, 1])
        )
    # print(f"TF matrix EEF -> CAM: \n{eef_to_cam_tf}")  # x forward, y left, z up

    bl_to_cam_tf = np.dot(bl_to_eef_tf, eef_to_cam_tf)

    # Convert object's camera coordinates to homogeneous form
    obj_cam_coords_h = np.array([*obj_cam_coords, 1])

    # Transform object's coordinates to base_link frame
    obj_bl_coords_h = np.dot(bl_to_cam_tf, obj_cam_coords_h)
    obj_bl_coords = obj_bl_coords_h[:3]  # Extract x, y, z from homogeneous coordinates
    # print(f"Object coordinates in base_link frame: {obj_bl_coords}")
    
    target = obj_bl_coords
    target[1] += 0.35  # offset in y (backwards)
    target[2] += 0.05  # offset in z (upwards)
    # obj_bl_coords = np.array([0.0, -0.3, 0.8])
    # print(f"Moving to object coordinates: {obj_bl_coords}")
    waypoint = np.array([0.0, -0.35, 0.2])
    robot.follow_eef_position_path_default_orientation(path=[waypoint, obj_bl_coords], wait=True)
    # rgb, depth= camera.read()
    # cv2.imwrite("latest_rgb.jpg",cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    # depth = np.clip(depth, 0, 1000)
    # cv2.imwrite("latest_depth.jpg", ((1 - depth / depth.max()) * 255).astype(np.uint8))
    time.sleep(1)
    robot.set_freedrive_mode(enable=True)
    init_xyz = robot.get_observation()["ee_pose_euler"][:3]
    rgb, depth = camera.read()

    key = None
    while key != "c":
        key = input("Press 'c' to continue: ")
        key = key.lower()

    target_xyz = robot.get_observation()["ee_pose_euler"][:3]

    robot.close_gripper()
    time.sleep(3)
    robot.set_freedrive_mode(enable=False)
    robot.go_to_eef_position_default_orientation(eef_position=init_xyz, wait=True)

    key = None
    while key != "y" and key != "d":
        key = input("Press 'y' to save, 'd' to discard: ")
        key = key.lower()

    if key == "y":
        sample_id = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        # Save initial images, position and target position
        cv2.imwrite(f"data/grasp_shelf/{sample_id}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(f"data/grasp_shelf/{sample_id}_depth.npy", depth)
        np.save(f"data/grasp_shelf/{sample_id}_init_xyz.npy", init_xyz)
        np.save(f"data/grasp_shelf/{sample_id}_target_xyz.npy", target_xyz)

    robot.open_gripper()
