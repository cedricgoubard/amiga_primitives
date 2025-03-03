import argparse
import time
import os
import glob

import cv2
import numpy as np
from omegaconf import OmegaConf



def Dev(cfg: OmegaConf):

    # This needs to be here to avoid unnecessary imports in the main script
    from amiga.vision import KitchenObjectDetector, overlay_results
    from amiga.drivers.cameras import ZEDCamera  # DO NOT REMOVE, used at eval
    from amiga.drivers.amiga import AMIGA  # DO NOT REMOVE, used at eval
    from amiga.primitives.handover import handover
    from amiga.primitives.grasp_shelf import grasp_from_shelf
    from amiga.models import GraspingLightningModule

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

    detector = KitchenObjectDetector(cfg.yolo_weights, time_buffer_sec=1.1)

    # Make robot client
    rob_backend = eval(cfg.robot_zmq.class_name)(cfg.robot_zmq)
    robot: AMIGA = rob_backend.make_zmq_client(cfg.robot_zmq.port, cfg.robot_zmq.host)

    grasp_mdl = GraspingLightningModule.load_from_checkpoint(cfg.grasp_mdl_ckpt_path)

    stop = False
    n_obj = 9
    already_grasped = []
    while not stop:
        try:
            grasped_obj = grasp_from_shelf(
                detector=detector, 
                camera=camera, 
                robot=robot,
                grasp_module=grasp_mdl,
                detect_obj_from_overlook=True,
                fall_back_after_grasp=False,
                already_grasped=already_grasped
            )

            already_grasped += [grasped_obj]

            # Construct path from grasp end 
            path = []
            blend = []

            fb_position = robot.get_observation()["ee_pose_euler"][:3]

            # First step is falling back from grasp
            fb_position[1] += 0.35
            fb_position[2] += 0.07
            path.append(fb_position)
            blend.append(0.1)

            # If we're below the counter, go through safe point first
            if fb_position[2] < 0.5:
                # First stop is a close safe point 
                path.append(robot.get_closest_safe_3d_position())
                blend.append(0.3)

                # Then add a high point
                path.append(np.array([0.0, -0.4, 0.8]))
                blend.append(0.25)
            
            handover(
                camera, 
                robot, 
                position=[0.0, -1.1, 0.7], 
                initial_path=path, 
                initial_blend=blend
                )


            n_obj -= 1 
            
            if n_obj == 0:
                stop = True

        except KeyboardInterrupt:
            stop = True
    