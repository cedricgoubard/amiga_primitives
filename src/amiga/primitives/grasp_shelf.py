import numpy as np

import transforms3d as t3d

from amiga.vision import KitchenObjectDetector, overlay_results
from amiga.drivers.cameras import ZEDCamera , CameraDriver, CameraParameters # DO NOT REMOVE, used at eval
from amiga.drivers.amiga import AMIGA  # DO NOT REMOVE, used at eval

def grasp_from_shelf(obj_name: str, detector: KitchenObjectDetector, camera: CameraDriver, robot: AMIGA):
    q = robot.get_named_joints_cfg(name="overlook")
    res = robot.go_to_joint_positions(joint_positions=q)  
    if res != True: raise ValueError("Failed to move to overlook position") 
    print(f"Moved to overlook position: {q}")

    rgb, depth = camera.read()
    objs = detector(rgb)
    olive_oil = [obj for obj in objs if obj["class_name"] == "olive-oil-bottle"]
    if len(olive_oil) == 0: raise ValueError("No olive oil bottle detected")
    print(f"Olive oil bottle detected: {olive_oil}")

    x, y, w, h = olive_oil[0]["xywh"]
    w, h = 0.9*w, 0.9*h  # make sure we get mostly the object
    depth_box = depth[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    obj_depth = np.median(depth_box)

    # Pixel space to 3D position in camera frame
    obj_cam_coords = camera.uvdepth_to_xyz(u=int(x), v=int(y), depth=obj_depth)
    print(f"Object coords in camera frame: {obj_cam_coords}")  # z forward, x up, y left

    eef_pose = robot.get_observation()["ee_pose_euler"]  # X Y Z rX rY rZ
    print(f"EEF pose in base_link: {eef_pose}")  # x left, y back, z up
    bl_to_eef_tf = t3d.affines.compose(
        T=np.array([eef_pose[0], eef_pose[1], eef_pose[2]]),
        R=t3d.euler.euler2mat(eef_pose[3], eef_pose[4], eef_pose[5], axes='sxyz'),
        Z=np.array([1, 1, 1])
        )
    print(f"TF matrix BL -> EEF: \n{bl_to_eef_tf}")  # x forward, y left, z up
    
    # cam_pose = np.array([0.091, 0.088, 0.03, -0.025, -1.315, -2.356])  # X Y Z rX rY rZ
    eef_to_cam_tf = t3d.affines.compose(
        T=np.array([0.091, 0.088, 0.03]),
        R=t3d.euler.euler2mat(0, 0, 3 * np.pi / 4, axes='sxyz'),
        Z=np.array([1, 1, 1])
        )
    print(f"TF matrix EEF -> CAM: \n{eef_to_cam_tf}")  # x forward, y left, z up

    bl_to_cam_tf = np.dot(bl_to_eef_tf, eef_to_cam_tf)

    # Convert object's camera coordinates to homogeneous form
    obj_cam_coords_h = np.array([*obj_cam_coords, 1])

    # Transform object's coordinates to base_link frame
    obj_bl_coords_h = np.dot(bl_to_cam_tf, obj_cam_coords_h)
    obj_bl_coords = obj_bl_coords_h[:3]  # Extract x, y, z from homogeneous coordinates
    print(f"Object coordinates in base_link frame: {obj_bl_coords}")

    # angles = [np.pi/2, -np.pi/4, 0.0]
    # robot.go_to_eef_pose(eef_pose=np.array([0.2, -0.6, 0.6, angles[0], angles[1], angles[2]]), gripper_position=1.0)
    # print(robot.get_observation()["ee_pose_euler"])



