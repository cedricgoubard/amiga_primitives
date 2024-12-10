import time
import datetime as dt
from typing import Union

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from einops import rearrange



class Rate:
    def __init__(self, rate: float):
        self._interval = 1.0 / rate
        self._next_time = time.time() + self._interval

    def sleep(self):
        now = time.time()
        sleep_time = self._next_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._next_time += self._interval


def centre_crop(rgb_frame, depth_frame):
    assert (
        len(rgb_frame.shape) == 3
        and rgb_frame.shape[-1] == 3 
        and len(depth_frame.shape) == 2
        and rgb_frame.shape[:-1] == depth_frame.shape
        ), "RGB frame should be (H, W, 3) and depth frame should be (H, W)"

    H, W = rgb_frame.shape[:-1]
    sq_size = min(H, W)

    h_start = (H - sq_size) // 2
    w_start = (W - sq_size) // 2

    # print(rgb_frame.shape, depth_frame.shape)

    rgb_frame = rgb_frame[h_start : h_start + sq_size, w_start : w_start + sq_size, :]
    depth_frame = depth_frame[h_start : h_start + sq_size, w_start : w_start + sq_size]

    return rgb_frame, depth_frame


def save_rgb(img, path: str = None, is_bgr: bool = False):
    import torch
    assert isinstance(img, (np.ndarray, torch.Tensor)), f"Image should be a numpy array or torch tensor, got {type(img)}"
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    assert len(img.shape) == 3, f"Image should be (H, W, C) or (C, H, W), got {img.shape}"

    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    if not is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if path is None:
        path = f"latest_rgb_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    cv2.imwrite(path, img)


def save_depth(depth, path: str = None, max=None):
    import torch
    assert isinstance(depth, (np.ndarray, torch.Tensor)), f"Depth should be a numpy array or torch tensor, got {type(depth)}"
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    assert (
        len(depth.shape) == 2
        or (len(depth.shape) == 3 and (depth.shape[0] == 1 or depth.shape[-1] == 1))
    ), f"Depth should be (H, W), (1, H, W) or (H, W, 1), got {depth.shape}"

    if max is not None:
        depth = np.clip(depth, 0, max)

    if path is None:
        path = f"latest_depth_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    cv2.imwrite(path, ((1 - depth / depth.max()) * 255).astype(np.uint8))


def create_grid_img(imgs):
    """Creates a grid of up to 3x3 images from a list of images."""
    if imgs.max() <= 1:
        imgs = np.clip((imgs * 255.0), 0, 255).astype(np.uint8)

    if imgs.shape[1] == 3:
        imgs = rearrange(imgs, 'b c h w -> b h w c')
    
    imgs = imgs[: min(9, len(imgs))]
    N = len(imgs)
    if N <= 3:
        fig, axes = plt.subplots(1, N, figsize=(3 * N, 3))
        axes = np.array([axes])
    else:
        rows = (N // 3) + (N % 3 > 0)
        fig, axes = plt.subplots(rows, 3, figsize=(9, rows * 3))
    for i, img in enumerate(imgs):
        axes[i // 3, i % 3].imshow(img)
        axes[i // 3, i % 3].axis("off")

    final_img = figure_to_image(fig)

    plt.close(fig)

    return final_img


def figure_to_image(fig: Figure) -> np.ndarray:
    """
    Converts a Matplotlib figure to a NumPy RGB array.

    Args:
        fig (Figure): The Matplotlib figure to convert.

    Returns:
        np.ndarray: The RGB image as a NumPy array with shape (H, W, 3).
    """
    # Attach a canvas to the figure and render it
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # Convert the rendered canvas to a string buffer and then to a NumPy array
    buf = canvas.buffer_rgba()
    image = np.asarray(buf, dtype=np.uint8)
    
    # Get the size of the figure in pixels
    w, h = canvas.get_width_height()
    
    # Reshape the flattened array into (H, W, 4) and extract RGB channels
    image = image.reshape((h, w, 4))
    rgb_image = image[:, :, :3]  # Drop the alpha channel
    return rgb_image
