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


    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    # Make robot client
    rob_backend = eval(cfg.zmq.class_name)(cfg.zmq)
    robot: AMIGA = rob_backend.make_zmq_client(cfg.zmq.port, cfg.zmq.host)
    
    if args.open:
        robot.open_gripper()
    elif args.close:
        robot.close_gripper()
    elif "freedrive" in args:
        robot.set_freedrive_mode(enable=args.freedrive)
    else:
        raise ValueError("No action specified")

