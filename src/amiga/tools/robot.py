import json
from datetime import datetime
import time
import argparse

import cv2
from omegaconf import OmegaConf
import numpy as np
import transforms3d as t3d

from amiga.drivers.amiga import AMIGA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file", required=True)
    parser.add_argument("--open", action="store_true", help="Open the gripper")
    parser.add_argument("--close", action="store_true", help="Close the gripper")
    parser.add_argument("--freedrive", action=argparse.BooleanOptionalAction, help="Set freedrive mode")
    parser.add_argument("--home", action="store_true", help="Go to home")


    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    # Make robot client
    rob_backend = eval(cfg.zmq.class_name)(cfg.zmq)
    robot: AMIGA = rob_backend.make_zmq_client(cfg.zmq.port, cfg.zmq.host)
    
    if args.open:
        robot.open_gripper()
    elif args.close:
        robot.close_gripper()
    elif args.freedrive is not None:
        robot.set_freedrive_mode(enable=args.freedrive)
    elif args.home:
        q = robot.get_named_joints_cfg(name="home")
        robot.go_to_joint_positions(joint_positions=q, wait=True)
    else:
        raise ValueError("No action specified")

