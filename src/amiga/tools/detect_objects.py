import time
import argparse

import cv2
from omegaconf import OmegaConf
import numpy as np

from amiga.drivers.cameras import ZEDCamera, RealsenseCamera  # DO NOT REMOVE
from amiga.vision import KitchenObjectDetector, overlay_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file", required=True)
    parser.add_argument("--weights", type=str, help="Path to the YOLO weights (.pt)", required=True)

    args = parser.parse_args()

    # Create Object detector
    mdl = KitchenObjectDetector(args.weights, time_buffer_sec=1.1)

    # Make camera client
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
    
    rate = 1
    print(f"Got image, looping at {rate} Hz")

    while True:
        rgb, _ = client.read()
        cv2.imwrite(f"latest_rgb.jpg", cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        t0 = time.time()
        res = mdl(rgb)
        t1 = time.time()
        # Model based on YOLOv11s -> 10-20ms per frame on RTX 3080 (amigo laptop)
        print(f"Time taken: {int((t1 - t0)*1000)}ms; results: {res}")

        # Overlay results
        rgb = overlay_results(rgb, res)
        cv2.imwrite(f"latest_objects.jpg", cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        time.sleep(1 / rate)
