from omegaconf import OmegaConf
import argparse

from amiga.drivers.amiga import AMIGA  # DO NOT REMOVE
from amiga.drivers.cameras import ZEDCamera, RealsenseCamera  # DO NOT REMOVE
from amiga.drivers.zmq import ZMQServer
from amiga.scripts import Dev  # DO NOT REMOVE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file", required=True)
    parser.add_argument("--zmq", action="store_true", help="Start a ZMQ component", required=False)
    parser.add_argument("--script", action="store_true", help="Start script", required=False)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    if args.zmq:
        backend = eval(cfg.zmq.class_name)(cfg)
        server = ZMQServer(backend, cfg.zmq.host, cfg.zmq.port)
        server.serve()

    elif args.script:
        eval(cfg.script_name)(cfg)
