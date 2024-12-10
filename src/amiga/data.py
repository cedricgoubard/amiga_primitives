import os
from typing import List

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class GraspingDataset(Dataset):
    def __init__(self, cfg, idcs: List[int] = None):
        """
        Initialises the dataset by scanning the directory for files and preparing
        a list of samples to load.

        Args:
            data_dir (str): Path to the data directory.
            idcs (List[int]): List of indices to load. If None, all samples are loaded.
        """
        self.data_dir = cfg.data_dir
        # Collect all the timestamps based on the shared prefix
        self.timestamps = sorted(set(f.split("_")[0] for f in os.listdir(cfg.data_dir) if "_" in f))

        if idcs is not None:
            self.timestamps = [self.timestamps[i] for i in idcs]

        self.cfg = cfg

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        """
        Loads and returns a single data sample.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            dict: Dictionary containing:
                - 'rgb': Tensor of shape (3, H, W) (RGB image).
                - 'depth': Tensor of shape (1, H, W) (Depth map).
                - 'dx_dy_dz': Tensor of shape (3,) (Target displacement).
        """
        timestamp = self.timestamps[idx]


        # Load RGB image
        rgb_path = os.path.join(self.data_dir, f"{timestamp}_rgb.png")
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        # Resize to 224x224
        rgb_image = cv2.resize(rgb_image, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_LINEAR)
        # Normalise to [0, 1]
        rgb_tensor = torch.tensor(rgb_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  

        # Load depth map
        depth_path = os.path.join(self.data_dir, f"{timestamp}_depth.npy")
        depth_data = np.load(depth_path)
        # Resize to 224x224
        depth_data = cv2.resize(depth_data, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_NEAREST)[..., None]
        # Normalise to [0, 1]; max depth is 50cm
        depth_tensor = torch.tensor(
            np.clip(depth_data, 0, self.cfg.max_depth_mm), dtype=torch.float32
            ).permute(2, 0, 1) / self.cfg.max_depth_mm


        # Load initial and target positions
        init_xyz_path = os.path.join(self.data_dir, f"{timestamp}_init_xyz.npy")
        target_xyz_path = os.path.join(self.data_dir, f"{timestamp}_target_xyz.npy")
        init_xyz = np.load(init_xyz_path)
        target_xyz = np.load(target_xyz_path)
        dx_dy_dz = torch.tensor(target_xyz - init_xyz, dtype=torch.float32)

        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "dx_dy_dz": dx_dy_dz,
        }

