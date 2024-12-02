import numpy as np

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

    print()

    x, y, w, h = olive_oil[0]["xywh"]
    w, h = 0.9*w, 0.9*h  # make sure we get mostly the object
    depth_box = depth[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    obj_depth = np.median(depth_box)

    # Pixel space to 3D position in camera frame
    x, y, z = camera.uvdepth_to_xyz(u=int(x), v=int(y), depth=obj_depth)

    print(f"Object coords in camera frame: {x, y, z}")

    eef_pose_in_bl = robot.get_observation()["ee_pose_euler"]  # X Y Z rX rY rZ
    eef_to_cam_tf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # X Y Z rX rY rZ
