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
    robot: AMIGA = rob_backend.make_zmq_client(cfg.robot_zmq.port, cfg.robot_zmq.host)

    stop = False
    while not stop:
        try:
            position = np.random.rand(3)
            position[0] = position[0] * 0.6 - 0.3
            position[1] = -1 * (position[1] * 0.1 + 0.5)
            position[2] = (position[2] * 0.9) - 0.2
            print(f"Moving to {position}")
            robot.go_to_eef_position_through_safe_point(eef_position=position, wait=True)
            handover(camera, robot)

            # Go back to high point
            position = np.array([0.0, -0.4, 0.8])
            robot.go_to_eef_position_default_orientation(eef_position=position, wait=True)

        except KeyboardInterrupt:
            stop = True
    