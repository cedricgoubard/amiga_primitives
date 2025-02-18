import time
import argparse

import cv2
from omegaconf import OmegaConf
import numpy as np

from amiga.drivers.cameras import ZEDCamera, RealsenseCamera  # DO NOT REMOVE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file", required=True)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)


    backend = eval(cfg.zmq.class_name)(cfg)
    client = backend.make_zmq_client(cfg.zmq.port, cfg.zmq.host, async_method="read")


    imgs = None
    print("Waiting for image...")   
    while imgs is None:
        imgs = client.read()
        if isinstance(imgs, dict) and "error" in imgs.keys():
            print("Error: ", imgs)
            imgs = None
            time.sleep(5)
        time.sleep(0.1)
    
    rate = 0.1
    print(f"Got image, looping at {rate} Hz")
    i = 0
    while True:
        rgb, depth = client.read()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"data/obj_det_v2/l{i}.png", rgb)
        i+=1

        # Normalise depth to 255
        # depth = np.clip(depth, 0, 2000)
        # depth = (depth.max() - depth) / depth.max() * 255
        # depth = depth.astype("uint8")
        # cv2.imwrite("latest_depth.jpg", depth)


        inp = input("Press enter to continue, q to quit")
        if inp == "q":
            break

        # time.sleep(1 / rate)
