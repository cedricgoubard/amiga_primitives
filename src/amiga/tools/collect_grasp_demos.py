import argparse
import time
import random

from omegaconf import OmegaConf
import numpy as np
import datetime as dt
import transforms3d as t3d
import cv2

from amiga.vision import KitchenObjectDetector, overlay_results
from amiga.drivers.cameras import ZEDCamera, CameraDriver  # DO NOT REMOVE, used at eval
from amiga.drivers.amiga import AMIGA  # DO NOT REMOVE, used at eval
from amiga.utils import save_rgb, save_depth
from amiga.models import GraspingLightningModule


def collect_grasp_demo(
        detector: KitchenObjectDetector, 
        camera: CameraDriver, 
        robot: AMIGA,
        obj_name: str = None, 
        grasp_module: GraspingLightningModule = None
        ):
    
    ############################ Move to overlook position #############################
    q = robot.get_named_joints_cfg(name="overlook")
    res = robot.go_to_joint_positions_through_safe_point(joint_positions=q, wait=True)  
    if res != True: raise ValueError("Failed to move to overlook position") 

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
    offset_y = random.uniform(0.35, 0.50)
    offset_z = random.uniform(0.05, 0.08)
    target[1] += offset_y  # offset in y (backwards)
    target[2] += offset_z  # offset in z (upwards)

    if target[2] < 0.4:
        wp = robot.get_named_eef_position(name="low_wp")
    else:
        wp = robot.get_named_eef_position(name="high_wp")
    

    robot.follow_eef_position_path_default_orientation(path=[wp, target], wait=True, blend=[0.1, 0.0])
    time.sleep(1.0)
    init_xyz = robot.get_observation()["ee_pose_euler"][:3]
    rgb_pre_grasp, depth_pre_grasp = camera.read()


    ################################### Collect demo ###################################
    if grasp_mdl is None:
        time.sleep(1)
        robot.set_freedrive_mode(enable=True, axes=[1, 1, 1, 0, 0, 0])

        key = None
        while key != "c":
            key = input("Press 'c' to continue: ")
            key = key.lower()

    else:
        rgb = cv2.resize(rgb_pre_grasp, (grasp_module.cfg.img_size, grasp_module.cfg.img_size))
        depth = cv2.resize(depth_pre_grasp, (grasp_module.cfg.img_size, grasp_module.cfg.img_size))

        rgb = (rgb / 255.0)
        depth = np.clip(depth, 0, grasp_module.cfg.max_depth_mm) / grasp_module.cfg.max_depth_mm

        res = grasp_module.pred_dx_dy_dz(rgb, depth)
        target_xyz = init_xyz + res
        target_xyz[1] += 0.06  # offset in y (backwards)
        target_xyz[2] += 0.04  # offset in z (upwards)

        robot.go_to_eef_position_default_orientation(eef_position=target_xyz, wait=True)
        

    ############################### Grasp and fall back ################################
    robot.close_gripper()
    time.sleep(2)
    if grasp_mdl is None: 
        target_xyz = robot.get_observation()["ee_pose_euler"][:3]
        robot.set_freedrive_mode(enable=False)
    fall_back_xyz = target_xyz.copy()
    fall_back_xyz[1] += 0.3
    fall_back_xyz[2] += 0.1

    # We will also record the opposite demo (placing on the shelf)
    # Since the object in the hand, we'll need to start from a tilted position
    default = t3d.euler.euler2quat(np.pi/2, -np.pi/4, 0.0, axes='sxyz')
    tilt_forward = t3d.euler.euler2quat(np.pi/12, -np.pi/12, 0.0, axes='rxyz')
    final_rot = t3d.quaternions.qmult(default, tilt_forward)
    rpy = t3d.euler.quat2euler(final_rot, axes='sxyz')
    
    fall_back_pose = np.append(fall_back_xyz, rpy)
    robot.go_to_eef_pose(eef_pose=fall_back_pose, wait=True)


    ######################## Record placing demo (opposite mvt) ########################
    rgb_pre_place, depth_pre_place = camera.read()
    save_rgb(rgb_pre_place, path="latest_rgb.jpg")
    save_depth(depth_pre_place, path="latest_depth.jpg", max=1000)
    
    key = None
    accepted_keys = ["y", "d"]
    msg = "Press 'y' to save, 'd' to discard"
    if grasp_mdl is not None: 
        accepted_keys += ["m"]
        msg += ", 'm' to modify"
    while key not in accepted_keys:
        key = input(msg + ": ")
        key = key.lower()
    
    if key == "y":
        sample_id = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        # Save initial images, position and target position
        save_rgb(rgb_pre_grasp, path=f"data/grasp_shelf_clean/{sample_id}_rgb.png")
        np.save(f"data/grasp_shelf_clean/{sample_id}_depth.npy", depth_pre_grasp)
        np.save(f"data/grasp_shelf_clean/{sample_id}_init_xyz.npy", init_xyz)
        np.save(f"data/grasp_shelf_clean/{sample_id}_target_xyz.npy", target_xyz)

        save_rgb(rgb_pre_place, path=f"data/place_shelf_clean/{sample_id}_rgb.png")
        np.save(f"data/place_shelf_clean/{sample_id}_depth.npy", depth_pre_place)
        np.save(f"data/place_shelf_clean/{sample_id}_init_xyz.npy", fall_back_xyz)
        np.save(f"data/place_shelf_clean/{sample_id}_target_xyz.npy", target_xyz)

    if key == "m":
        print("Current target position: ", target_xyz)
        target_xyz = np.array([float(x) for x in input("Enter corrected target position (x y z): ").split()])
        print("Saving target position: ", target_xyz)
        sample_id = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        # Save initial images, position and target position
        save_rgb(rgb_pre_grasp, path=f"data/grasp_shelf_clean/{sample_id}_rgb.png")
        np.save(f"data/grasp_shelf_clean/{sample_id}_depth.npy", depth_pre_grasp)
        np.save(f"data/grasp_shelf_clean/{sample_id}_init_xyz.npy", init_xyz)
        np.save(f"data/grasp_shelf_clean/{sample_id}_target_xyz.npy", target_xyz)

        save_rgb(rgb_pre_place, path=f"data/place_shelf_clean/{sample_id}_rgb.png")
        np.save(f"data/place_shelf_clean/{sample_id}_depth.npy", depth_pre_place)
        np.save(f"data/place_shelf_clean/{sample_id}_init_xyz.npy", fall_back_xyz)
        np.save(f"data/place_shelf_clean/{sample_id}_target_xyz.npy", target_xyz)

    robot.go_to_eef_position_default_orientation(eef_position=target_xyz, wait=True)
    robot.open_gripper()
    time.sleep(1.0)
    robot.go_to_eef_position_default_orientation(eef_position=fall_back_xyz, wait=True)



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

    grasp_mdl = None
    if "grasp_mdl_ckpt_path" in cfg.keys() and cfg.grasp_mdl_ckpt_path is not None:
        # Load grasping model
        grasp_mdl = GraspingLightningModule.load_from_checkpoint(cfg.grasp_mdl_ckpt_path)

    stop = False
    while not stop:
        try:
            collect_grasp_demo(mdl, camera, robot, grasp_module=grasp_mdl)
        except KeyboardInterrupt:
            stop = True
    