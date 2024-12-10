import argparse
import time
import random

from omegaconf import OmegaConf
import numpy as np
import datetime as dt
import transforms3d as t3d

from amiga.vision import KitchenObjectDetector, overlay_results
from amiga.drivers.cameras import ZEDCamera, CameraDriver  # DO NOT REMOVE, used at eval
from amiga.drivers.amiga import AMIGA  # DO NOT REMOVE, used at eval
from amiga.utils import save_rgb, save_depth


def collect_grasp_demo(
        detector: KitchenObjectDetector, 
        camera: CameraDriver, 
        robot: AMIGA,
        obj_name: str = None, 
        ):
    
    ############################ Move to overlook position #############################
    q = robot.get_named_joints_cfg(name="overlook")
    res = robot.go_to_joint_positions_through_safe_point(joint_positions=q, wait=True)  
    if res != True: raise ValueError("Failed to move to overlook position") 

    key = None
    while key != "c":
        key = input("Press 'c' to continue: ")
        key = key.lower()


    ################################## Detect object ###################################
    rgb, depth = camera.read()
    objs = detector(rgb, tracking=False)
    
    # To collect data for obj det
    # sample_path = f'data/obj_det/{dt.datetime.now().strftime("%Y%m%d%H%M%S")}_rgb.png'
    # save_rgb(rgb, path=sample_path)
    
    # For debug
    save_rgb(rgb, path="latest_rgb.jpg")
    save_depth(depth, path="latest_depth.jpg", max=3000)
    if len(objs) > 0: 
        save_rgb(overlay_results(rgb, objs), path="latest_objects.jpg")

    if obj_name is None:
        det_obj = [random.choice(objs)] 
    else:
        det_obj = [obj for obj in objs if obj["class_name"] == obj_name]
        if len(det_obj) == 0: raise ValueError(f"No {obj_name} bottle detected")

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
    
    init_xyz = robot.get_observation()["ee_pose_euler"][:3]
    rgb, depth = camera.read()


    ################################### Collect demo ###################################
    time.sleep(1)
    robot.set_freedrive_mode(enable=True)

    key = None
    while key != "c":
        key = input("Press 'c' to continue: ")
        key = key.lower()

    target_xyz = robot.get_observation()["ee_pose_euler"][:3]


    ############################### Grasp and fall back ################################
    robot.close_gripper()
    time.sleep(2)
    robot.set_freedrive_mode(enable=False)
    fall_back_xyz = init_xyz
    fall_back_xyz[1] += 0.2
    fall_back_xyz[2] += 0.09

    # We will also record the opposite demo (placing on the shelf)
    # Since the object in the hand, we'll need to start from a tilted position
    default = t3d.euler.euler2quat(np.pi/2, -np.pi/4, 0.0, axes='sxyz')
    tilt_forward = t3d.euler.euler2quat(np.pi/12, -np.pi/12, 0.0, axes='rxyz')
    final_rot = t3d.quaternions.qmult(default, tilt_forward)
    rpy = t3d.euler.quat2euler(final_rot, axes='sxyz')
    
    fall_back_pose = np.append(fall_back_xyz, rpy)
    robot.go_to_eef_pose(eef_pose=fall_back_pose, wait=True)


    ######################## Record placing demo (opposite mvt) ########################
    rgb, depth = camera.read()
    save_rgb(rgb, path="latest_rgb.jpg")
    save_depth(depth, path="latest_depth.jpg", max=1000)
    
    key = None
    while key != "y" and key != "d":
        key = input("Press 'y' to save, 'd' to discard: ")
        key = key.lower()
    
    if key == "y":
        pass
        sample_id = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        # Save initial images, position and target position
        save_rgb(rgb, path=f"data/grasp_shelf/{sample_id}_rgb.png")
        np.save(f"data/grasp_shelf/{sample_id}_depth.npy", depth)
        np.save(f"data/grasp_shelf/{sample_id}_init_xyz.npy", init_xyz)
        np.save(f"data/grasp_shelf/{sample_id}_target_xyz.npy", target_xyz)

        save_rgb(rgb, path=f"data/put_on_shelf/{sample_id}_rgb.png")
        np.save(f"data/put_on_shelf/{sample_id}_depth.npy", depth)
        np.save(f"data/put_on_shelf/{sample_id}_init_xyz.npy", init_xyz)
        np.save(f"data/put_on_shelf/{sample_id}_target_xyz.npy", target_xyz)

    robot.go_to_eef_position_default_orientation(eef_position=target_xyz, wait=True)
    robot.open_gripper()
    time.sleep(1.0)
    robot.go_to_eef_position_default_orientation(eef_position=init_xyz, wait=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file", required=True)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    # Create Object detector
    mdl = KitchenObjectDetector(cfg.yolo_weights, time_buffer_sec=1.1)

    # Make camera client
    cam_backend = eval(cfg.cam_zmq.class_name)(cfg.cam_zmq)
    camera = cam_backend.make_zmq_client(cfg.cam_zmq.port, cfg.cam_zmq.host, async_method="read")

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

    # Make robot client
    rob_backend = eval(cfg.robot_zmq.class_name)(cfg.robot_zmq)
    robot = rob_backend.make_zmq_client(cfg.robot_zmq.port, cfg.robot_zmq.host)

    stop = False
    while not stop:
        try:
            collect_grasp_demo(mdl, camera, robot)
        except KeyboardInterrupt:
            stop = True
    