import os
import time
from typing import Dict, List, Optional, Tuple
from abc import abstractmethod

import numpy as np
import cv2
from jaxtyping import Float, Int

from amiga.drivers.zmq import ZMQBackendObject
from amiga.utils import centre_crop


def check_zed_required_libraries() -> bool:
    ready = True
    try:
        import pyzed.sl
    except ImportError:
        ready = False
    return ready

def check_realsense_required_libraries() -> bool:
    ready = True
    try:
        import pyrealsense2 as rs
    except ImportError:
        ready = False
    return ready


def get_zed_resolution(h: int, w: int):
    check_zed_required_libraries()
    import pyzed.sl as sl
    #             width*height
    # | HD2K    | 2208*1242 (x2) \n Available FPS: 15 |
    # | HD1080  | 1920*1080 (x2) \n Available FPS: 15, 30 |
    # | HD1200  | 1920*1200 (x2) \n Available FPS: 15, 30, 60 |
    # | HD720   | 1280*720 (x2) \n Available FPS: 15, 30, 60 |
    # | VGA     | 672*376 (x2) \n Available FPS: 15, 30, 60, 100 |

    if h <= 376 and w <= 672:
        return sl.RESOLUTION.VGA
    elif h <= 720 and w <= 1280:
        return sl.RESOLUTION.HD720
    elif h <= 1080 and w <= 1920:
        return sl.RESOLUTION.HD1080
    elif h <= 1200 and w <= 1920:
        return sl.RESOLUTION.HD1200
    else:
        return sl.RESOLUTION.HD2K


def get_device_ids() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


def get_d4X5_resolution_colour(h: int, w: int):
    import pyrealsense2 as rs
    
    available_resolutions = [
        (1920, 1080), (1280, 720), (960, 540), (848, 480), (640, 480), (640, 360), 
        (424, 240), (320, 240), (320, 180)
    ]

    for res in available_resolutions[::-1]:
        if h <= res[1] and w <= res[0]:
            return res
    
    return available_resolutions[-1]


def get_d4X5_resolution_depth(h: int, w: int):
    import pyrealsense2 as rs
    
    available_resolutions = [
        (1280, 720), (848, 480), (640, 480), (640, 360), (480, 270), (424, 240), 
        (256, 144)
    ]

    for res in available_resolutions[::-1]:
        if h <= res[1] and w <= res[0]:
            return res
    
    return available_resolutions[-1]


class CameraParameters(object):
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    def __str__(self) -> str:
        return f"CamParams(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, k1={self.k1}, k2={self.k2}, p1={self.p1}, p2={self.p2}, k3={self.k3})"


class CameraDriver(ZMQBackendObject):
    """Camera protocol.

    A protocol for a camera driver. This is used to abstract the camera from the rest of the code.
    """

    def __init__(self, cfg) -> None:
        pass

    @abstractmethod
    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Int[np.ndarray, "H W 3"],  Float[np.ndarray, "H W 1"]]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.

        Returns:
            np.ndarray: The color image.
            np.ndarray: The depth image.
        """
        raise NotImplementedError()

    def _postprocess_img(
            self, 
            colour_image: Int[np.ndarray, "H W 3"], 
            depth_image: Float[np.ndarray, "H W"], 
            img_size: Optional[Tuple[int, int]] = None
        )-> Tuple[Int[np.ndarray, "H W 3"], Float[np.ndarray, "H W 1"]]:
        """
        Post-processes the given colour and depth images.
        This function performs several operations on the input images, including
        center cropping, setting NaN and infinite values to 0 in the depth image,
        resizing the images if a target size is provided, and optionally flipping
        the images 180 degrees. 
        Args:
            colour_image (Int[np.ndarray, "H W 3"]): The input colour image.
            depth_image (Float[np.ndarray, "H W 1"]): The input depth image.
            img_size (Optional[Tuple[int, int]], optional): The target size for resizing the images. Defaults to None.
        Returns:
            Tuple[Int[np.ndarray, "H W 3"], Float[np.ndarray, "H W 1"]]: The processed colour and depth images.
        """
        colour_image, depth_image = centre_crop(
            colour_image,
            depth_image
            )
                
        depth_image = np.nan_to_num(depth_image, nan=50000, posinf=50000, neginf=0)

        if img_size is None:
            image = colour_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(colour_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self.flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            depth = depth[:, :, None]

        return image, depth

    @abstractmethod
    def get_parameters(self) -> CameraParameters:
        """Get the camera parameters.

        Returns:
            CameraParameters: The camera parameters.
        """
        raise NotImplementedError()

    def uvdepth_to_xyz(self, u: int, v: int, depth: float) -> Tuple[float, float, float]:
        """u, v are pixel coordinates, depth is in mm. Result coords should be in meters."""
        params = self.get_parameters()
        fx, fy, cx, cy = params.fx, params.fy, params.cx, params.cy

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return x / 1000, y / 1000, z / 1000

    def get_methods(self) -> Dict[str, str]:
        return {
            "read": ["img_size"],
            "get_parameters": [],
            "uvdepth_to_xyz": ["u", "v", "depth"]
        }


class ZEDCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"ZEDCamera(device_id={self._device_id}, resolution={self._resolution})"

    def __init__(self, cam_cfg):
        has_driver = check_zed_required_libraries()

        if not has_driver:
            # This class might be instantiated by a client, just to know its methods; so no error here
            return
        
        import pyzed.sl as sl

        self._zed = sl.Camera()
        self._device_id = cam_cfg.sn
        self._min_depth = cam_cfg.min_depth
        self._max_depth = cam_cfg.max_depth
        self.flip = False  # Handled below; this is used in postprocess

        self._resolution = get_zed_resolution(cam_cfg.rgb_res.height, cam_cfg.rgb_res.width)

        # see https://github.com/stereolabs/zed-python-api/blob/master/src/pyzed/sl_c.pxd
        init_params = sl.InitParameters()
        init_params.sdk_verbose = 0
        init_params.camera_resolution = self._resolution
        
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # SPEED, PERFORMANCE, ULTRA, NEURAL_PLUS
        init_params.depth_minimum_distance = self._min_depth
        init_params.depth_maximum_distance = self._max_depth
        init_params.camera_fps = 30
        if cam_cfg.flip: init_params.camera_image_flip = sl.FLIP_MODE.ON
        else: init_params.camera_image_flip = sl.FLIP_MODE.OFF

        # This needs to be high, otherwise we get CAMERA_NOT_FOUND or LOW_USB_BANDWIDTH
        init_params.open_timeout_sec = 20

        # Open the camera
        n_tries, success = 10, False
        print("Initialising ZED camera...")
        while not success and n_tries > 0:
            t0 = time.time()
            err = self._zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"Fail; will retry {n_tries} more times -", repr(err))
                n_tries -= 1
            else:
                print(f"ZED camera initialised in {round(time.time() - t0, 1)}s.")
                success = True

        if not success:
            self._zed.close()
            raise RuntimeError("Failed to open ZED camera.")

    def get_parameters(self) -> CameraParameters:
        import pyzed.sl as sl

        cam_params = self._zed.get_camera_information().camera_configuration.calibration_parameters
        params = CameraParameters()
        params.fx = cam_params.left_cam.fx
        params.fy = cam_params.left_cam.fy
        params.cx = cam_params.left_cam.cx
        params.cy = cam_params.left_cam.cy
        params.k1 = cam_params.left_cam.disto[0]
        params.k2 = cam_params.left_cam.disto[1]
        params.p1 = cam_params.left_cam.disto[2]
        params.p2 = cam_params.left_cam.disto[3]
        params.k3 = cam_params.left_cam.disto[4]

        return params

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Int[np.ndarray, "H W 3"],  Float[np.ndarray, "H W 1"]]:

        import pyzed.sl as sl

        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.enable_depth = True
        runtime_parameters.enable_fill_mode = False

        image = sl.Mat()
        depth = sl.Mat()

        while self._zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.1)

        self._zed.retrieve_image(image, sl.VIEW.LEFT)
        self._zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        colour_image = image.get_data()  # HxWx4, 4th is alpha
        colour_image = colour_image[:, :, :3]  # HxWx3
        depth_image = depth.get_data()  # HxW

        image, depth = self._postprocess_img(colour_image, depth_image, img_size=img_size)

        return image, depth


class RealsenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id}, RGB resolution={self.rgb_res}), Depth resolution={self.d_res})"

    def __init__(self, cfg):
        has_driver = check_realsense_required_libraries()
        if not has_driver:
            # This class might be instantiated by a client, just to know its methods; so no error here
            return

        import pyrealsense2 as rs

        self._device_id = None
        if cfg.sn is not None: self._device_id = cfg.sn

        self.rgb_res = (320, 240)
        if cfg.rgb_res is not None: self.rgb_res = get_d4X5_resolution_colour(cfg.rgb_res.height, cfg.rgb_res.width)

        self.d_res = (424, 240)
        if cfg.d_res is not None: self.d_res = get_d4X5_resolution_depth(cfg.d_res.height, cfg.d_res.width)

        self.flip = False
        if cfg.flip is not None: self.flip = cfg.flip

        self._max_depth = 2.0
        if cfg.max_depth is not None: self._max_depth = cfg.max_depth

        self._min_depth = 0.1
        if cfg.min_depth is not None: self._min_depth = cfg.min_depth

        if self._device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self._pipeline = rs.pipeline()
            config = rs.config()
        else:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(str(self._device_id))

        # use rs-enumerate-devices to list resolutions
        # installed with vcpkg install realsense2[core,tools]
        config.enable_stream(rs.stream.depth, self.d_res[0], self.d_res[1], rs.format.z16, 30)  # Could be 60    424, 240
        config.enable_stream(rs.stream.color, self.rgb_res[0], self.rgb_res[1], rs.format.bgr8, 30)  # Could be 60   320, 240,
        self._pipeline.start(config)

        print(self)

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Int[np.ndarray, "H W 3"],  Float[np.ndarray, "H W 1"]]:
       
        frames = self._pipeline.wait_for_frames()
        colour_frame = frames.get_color_frame()
        colour_image = np.asanyarray(colour_frame.get_data())
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())

        image, depth = self._postprocess_img(colour_image, depth_image, img_size=img_size)

        return image, depth