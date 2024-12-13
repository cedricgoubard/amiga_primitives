import argparse
import time

import cv2
from omegaconf import OmegaConf



def Dev(cfg: OmegaConf):

    # This needs to be here to avoid unnecessary imports in the main script
    from amiga.vision import KitchenObjectDetector, overlay_results
    from amiga.drivers.cameras import ZEDCamera  # DO NOT REMOVE, used at eval
    from amiga.drivers.amiga import AMIGA  # DO NOT REMOVE, used at eval
    from amiga.primitives.grasp_shelf import grasp_from_shelf
    from amiga.models import GraspingLightningModule

    # Create Object detector
    obj_det_mdl = KitchenObjectDetector(cfg.yolo_weights, time_buffer_sec=1.1)

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

    # Load grasping model
    grasp_mdl = GraspingLightningModule.load_from_checkpoint(cfg.grasp_mdl_ckpt_path)

    stop = False
    while not stop:
        try:
            grasp_from_shelf(
                detector=obj_det_mdl,
                camera=camera, 
                robot=robot, 
                grasp_module=grasp_mdl
                )
        except KeyboardInterrupt:
            stop = True
    