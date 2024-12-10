from typing import Tuple

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.v2 as T
import pytorch_lightning as L
import wandb
from jaxtyping import Float

from amiga.utils import create_grid_img


def compute_l2_distance(
        pred: Float[torch.Tensor, "B 3"], 
        target: Float[torch.Tensor, "B 3"]
        ) -> Tuple[Float[torch.Tensor, "B"], Float[torch.Tensor, "B"], Float[torch.Tensor, "B"]]:
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)

    dist = torch.cdist(pred, target, p=2)
    x_dist = torch.cdist(pred[:, :, 0], target[:, :, 0], p=2)
    y_dist = torch.cdist(pred[:, :, 1], target[:, :, 1], p=2)
    z_dist = torch.cdist(pred[:, :, 2], target[:, :, 2], p=2)

    return dist.squeeze(1).squeeze(1), x_dist.squeeze(1), y_dist.squeeze(1), z_dist.squeeze(1)


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

            dist, x_dist, y_dist, z_dist = compute_l2_distance(pred_unnorm, target_unnorm)
            dist_mean = dist.mean()
            dist_std = dist.std()

            self.log(f"{step}_dist_mm_mean", dist_mean*1000, prog_bar=(step == "val"))
            self.log(f"{step}_dist_mm_std", dist_std*1000)

            for coord, dist in zip(["x", "y", "z"], [x_dist, y_dist, z_dist]):                
                self.log(f"{step}_dist_{coord}_mm_std", dist.std()*1000)
                self.log(f"{step}_dist_{coord}_mm_mean", dist.mean()*1000)
                
        return mse_loss

    def configure_optimizers(self):
        """
        Configure optimizers for training.
        """
        print(f"Using optimizer {self.cfg.optimizer}")
        if self.cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate, momentum=0.9)
        elif self.cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate, weight_decay=0.01)
        else: 
            raise ValueError(f"Unknown optimizer {self.cfg.optimizer}")
        return optimizer

