import os
import argparse
from typing import List
from datetime import datetime

from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.v2 as T
import numpy as np
import cv2
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import RichProgressBar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from einops import rearrange
import wandb


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


class SmallCNNBackbone(nn.Module):
    def __init__(self, img_size: int):
        super(SmallCNNBackbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * (img_size // (4*2*2*2)) ** 2, 1024)
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=4, stride=4)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


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


class GraspingModel(nn.Module):
    def __init__(self, cfg):
        super(GraspingModel, self).__init__()
        if cfg.use_rgb:
            if cfg.rgb_backbone == "resnet18":
                # Load a pre-trained ResNet18 backbone for RGB
                self.rgb_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                self.rgb_backbone.fc = nn.Identity()  # Remove final classification layer
            elif cfg.rgb_backbone == "smallcnn":
                self.rgb_backbone = SmallCNNBackbone(cfg.img_size)
            else:
                raise ValueError(f"Unknown RGB backbone {cfg.rgb_backbone}")
            
            if cfg.freeze_rgb_backbone:
                for param in self.rgb_backbone.parameters():
                    param.requires_grad = False
            # Print n params
            print(f"RGB backbone ({cfg.rgb_backbone}): {sum(p.numel() for p in self.rgb_backbone.parameters()) / 1e6:.1f}M ({sum(p.numel() for p in self.rgb_backbone.parameters() if p.requires_grad) / 1e6:.1f}M trainable)")

        if cfg.use_depth:
            # Load an adapted resnet for depth
            if cfg.depth_backbone == "resnet18":
                self.depth_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                self.depth_backbone.fc = nn.Identity()  # Remove final classification layer
                self.depth_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif cfg.depth_backbone == "smallcnn":
                self.depth_backbone = SmallCNNBackbone(cfg.img_size)
            else: 
                raise ValueError(f"Unknown depth backbone {cfg.depth_backbone}")

            if cfg.freeze_depth_backbone:
                for param in self.depth_backbone.parameters():
                    param.requires_grad = False
            # Print n params
            print(f"Depth backbone ({cfg.depth_backbone}): {sum(p.numel() for p in self.depth_backbone.parameters()) / 1e6:.1f}M ({sum(p.numel() for p in self.depth_backbone.parameters() if p.requires_grad) / 1e6:.1f}M trainable)")

        # Combine processed RGB and depth features and predict dx, dy, dz
        first_layer_input = (
            (512 if cfg.use_rgb else 0)
            + (512 if cfg.use_depth else 0)
        )
        print(f"First layer input: {first_layer_input}")
        self.fc = nn.Sequential(
            nn.Linear(first_layer_input, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 3)  # Output dx, dy, dz
        )
        # Print n params
        print(f"Total: {sum(p.numel() for p in self.fc.parameters()) / 1e6:.1f}M ({sum(p.numel() for p in self.fc.parameters() if p.requires_grad) / 1e6:.1f}M trainable)")

        self.cfg = cfg

    def forward(self, rgb, depth):
        # Process RGB through ResNet18
        if self.cfg.use_rgb:
            rgb_features = self.rgb_backbone(rgb)

        # Process depth through a small convolutional network
        if self.cfg.use_depth:
            depth_features = self.depth_backbone(depth)

        if self.cfg.use_rgb and self.cfg.use_depth:
            # Concatenate the features and pass through fully connected layers
            combined_features = torch.cat((rgb_features, depth_features), dim=1)
        elif self.cfg.use_rgb:
            combined_features = rgb_features
        elif self.cfg.use_depth:
            combined_features = depth_features
        
        output = self.fc(combined_features)
        return output


class GraspingLightningModule(L.LightningModule):
    def __init__(self, cfg: OmegaConf, stats=None):
        super(GraspingLightningModule, self).__init__()
        if stats is None: 
            print("Stats not provided, using default values")
            stats = {
                'depth': {'mean': torch.tensor(0.8479), 'std': torch.tensor(0.2752)}, 
                'dx_dy_dz': {'mean': torch.tensor([-0.0279, -0.1569, -0.0450]), 'std': torch.tensor([0.0127, 0.0309, 0.0187])}
                }
        self.stats = stats
       
        self.model = GraspingModel(cfg)
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        self.augments = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.15, saturation=0.15, hue=0.15),
                T.RandomAdjustSharpness(sharpness_factor=0.3),
                T.GaussianNoise(mean=0, sigma=0.12),
            ])
        
        self._train_sample_is_saved = False
        self._val_sample_is_saved = False

    def _save_grid(self, batch, label):
        # Save up to 9 images from the batch in a single grid image for visualisation 
        selected_imgs = batch["rgb"][:min(9, batch["rgb"].shape[0])]
        selected_imgs = selected_imgs.cpu().numpy()

        # print(f"{dt.datetime.now().strftime(r'%Y-%m-%dT%H:%M:%S.%f')} - Saving image {label} {i}")
        img = create_grid_img(selected_imgs)
        self.logger.log_image(key=f"input_rgb_"+label, images=[img])
        # print(f"{dt.datetime.now().strftime(r'%Y-%m-%dT%H:%M:%S.%f')} - Image {label} {i} saved")

    def forward(self, rgb, depth):
        """
        Forward pass of the model.
        """
        # Normalise depth images
        depth = (depth - self.stats["depth"]["mean"].to(self.device)) / self.stats["depth"]["std"].to(self.device)
        # Normalise images with ImageNet statistics
        means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        rgb = (rgb - means) / stds
        return self.model(rgb, depth)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        """
        batch["rgb"] = self.augments(batch["rgb"])

        if not self._train_sample_is_saved:
            self._save_grid(batch, "train")
            self._train_sample_is_saved = True

            # Log image sizes, min and max values
            for key, value in batch.items():
                if "rgb" in key or "depth" in key:
                    wandb.log({
                        f"image_sizes_{key}": value.shape,
                        f"image_min_{key}": value.min(),
                        f"image_max_{key}": value.max(),
                    })

        loss = self.compute_loss(batch, step="train")
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        if not self._val_sample_is_saved:
            self._save_grid(batch, "val")
            self._val_sample_is_saved = True

        loss = self.compute_loss(batch, step="val")

    def compute_loss(self, batch, step):
        rgb, depth, target = batch["rgb"], batch["depth"], batch["dx_dy_dz"]
        predictions = self.forward(rgb, depth)

        # Normalise target values
        target = (target - self.stats["dx_dy_dz"]["mean"].to(self.device)) / self.stats["dx_dy_dz"]["std"].to(self.device)

        mse_loss = F.mse_loss(predictions, target)
        self.log(f"{step}_loss", mse_loss, prog_bar=(step == "val"))
        
        if step == "val":
            # Unnormalise for logging
            pred_unnorm = predictions * self.stats["dx_dy_dz"]["std"].to(self.device) + self.stats["dx_dy_dz"]["mean"].to(self.device)
            target_unnorm = target * self.stats["dx_dy_dz"]["std"].to(self.device) + self.stats["dx_dy_dz"]["mean"].to(self.device)

            pred_unnorm = pred_unnorm.unsqueeze(1)
            target_unnorm = target_unnorm.unsqueeze(1)

            dist = torch.cdist(pred_unnorm, target_unnorm, p=2)
            dist_mean = dist.mean()
            dist_std = dist.std()

            self.log(f"{step}_dist_mm_mean", dist_mean*1000, prog_bar=(step == "val"))
            self.log(f"{step}_dist_mm_std", dist_std*1000)

            for i, coord in enumerate(["x", "y", "z"]):
                coord_dist = torch.cdist(pred_unnorm[:, :, i], target_unnorm[:, :, i], p=2)
                
                self.log(f"{step}_dist_{coord}_mm_std", coord_dist.std()*1000)
                self.log(f"{step}_dist_{coord}_mm_mean", coord_dist.mean()*1000)
                
        return mse_loss

    def configure_optimizers(self):
        """
        Configure optimizers for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        return optimizer


def update_cfg_with_sweep_params(cfg):
    """
    Updates the config with sweep parameters.
    """
    sweep_config = wandb.config
    cfg.img_size = sweep_config.img_size
    cfg.max_depth_mm = sweep_config.max_depth_mm
    cfg.use_rgb = sweep_config.use_rgb
    cfg.depth_backbone = sweep_config.depth_backbone
    cfg.learning_rate = sweep_config.learning_rate
    cfg.batch_size_train = sweep_config.batch_size_train
    return cfg


def main(cfg, is_sweep: bool = False):
    L.seed_everything(cfg.seed)

    wandb.init()
    if is_sweep:
        cfg = update_cfg_with_sweep_params(cfg)
        
    n_samples = len(set(f.split("_")[0] for f in os.listdir(cfg.data_dir) if "_" in f))
    
    shuffled_indices = np.random.permutation(n_samples)
    train_indices = shuffled_indices[: int(cfg.train_ratio * n_samples)]
    val_indices = shuffled_indices[int(cfg.train_ratio * n_samples) :]
    
    train_dataset = GraspingDataset(cfg, idcs=train_indices)
    val_dataset = GraspingDataset(cfg, idcs=val_indices)
    print(f"{len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    
    # Iterate over all dataset, compute mean and std of depth image
    stats = {"depth": {"mean": 0, "std": 0}, "dx_dy_dz": {"mean": [], "std": 0, }}
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        stats["depth"]["mean"] += sample["depth"].mean()
        stats["depth"]["std"] += sample["depth"].std()
        stats["dx_dy_dz"]["mean"] += [sample["dx_dy_dz"]]

    stats["depth"]["mean"] /= len(train_dataset)
    stats["depth"]["std"] /= len(train_dataset)
    stats["dx_dy_dz"]["std"] = torch.stack(stats["dx_dy_dz"]["mean"]).std(dim=0)
    stats["dx_dy_dz"]["mean"] = torch.stack(stats["dx_dy_dz"]["mean"]).mean(dim=0)
   
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        # prefetch_factor=1,
        # persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        # prefetch_factor=1,
        # persistent_workers=True,
    )

    model = GraspingLightningModule(
        cfg=cfg,
        stats=stats,
        )

    callbacks = [RichProgressBar()]
    if "mdl_ckpt" in cfg.keys():
        callbacks.append(ModelCheckpoint(**dict(cfg.mdl_ckpt)))
    if "early_stopping" in cfg.keys():
        callbacks.append(EarlyStopping(**dict(cfg.early_stopping)))

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        logger=WandbLogger(**dict(cfg.wandb)),
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file", required=True)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    current_time = {"now": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}
    cfg = OmegaConf.merge(cfg, OmegaConf.create(current_time))
    os.makedirs(cfg.logs_dir, exist_ok=True)

    torch.set_float32_matmul_precision('high')  # medium or high
    torch.multiprocessing.set_start_method('spawn')

    # For simple exec
    # main(cfg, is_sweep=False)

    # For sweeping
    wandb.login()
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "img_size": {"values": [224, 480, 640]},
            "max_depth_mm": {"values": [400, 600, 700]},
            "use_rgb": {"values": [True, False]},
            "depth_backbone": {"values": ["resnet18", "smallcnn"]},
            "learning_rate": {"min": 1e-5, "max": 1e-2},
            "batch_size_train": {"values": [16, 32, 48]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="amigrasp")
    wandb.agent(sweep_id, function=lambda: main(cfg, is_sweep=True), count=500)
    