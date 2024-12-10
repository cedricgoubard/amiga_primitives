import os
import argparse
from datetime import datetime

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import RichProgressBar
import wandb

from amiga.models import GraspingLightningModule
from amiga.data import GraspingDataset


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
    cfg.optimizer = sweep_config.optimizer
    return cfg


def main(cfg, is_sweep: bool = False):
    L.seed_everything(cfg.seed)

    wandb.init()
    if is_sweep:
        cfg = update_cfg_with_sweep_params(cfg)

    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("val_dist_mm_mean", summary="min")
        
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
            "img_size": {"values": [224, 480]},
            "max_depth_mm": {"values": [400, 600, 700]},
            "use_rgb": {"values": [True, False]},
            "depth_backbone": {"values": ["resnet18", "smallcnn"]},
            "learning_rate": {"values": [1e-3, 1e-4, 1e-5]},
            "batch_size_train": {"values": [16, 32, 48]},
            "optimizer": {"values": ["adam", "adamw", "sgd"]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="amigrasp")
    wandb.agent(sweep_id, function=lambda: main(cfg, is_sweep=True), count=500)
    