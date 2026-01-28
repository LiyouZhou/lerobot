import json
import math
import os
from dataclasses import dataclass, asdict, field
from itertools import cycle
from pathlib import Path
from safetensors.torch import save_file

import hydra
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from einops import rearrange
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm, trange

import wandb
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodicBatchSampler
from lerobot.policies.smolvla.memory.ViTMemory import (
    DINOv2wMemory,
    ViTwMemory,
    ViTMemoryConfig,
    normalize,
    unnormalize,
)
from lerobot.utils.utils import print_cuda_memory_usage
from datetime import datetime
import secrets
import string


def prevent_tf_gpu_memory_grab():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_dataset(ds_name, data_dir, action_dim):
    ds, info = tfds.load(
        ds_name, split="train", data_dir=data_dir, with_info=True, download=False
    )

    val_ds = tfds.load(ds_name, split="val", data_dir=data_dir, download=False)

    metadata = {
        "action": {
            "max": np.array([-math.inf] * action_dim),
            "min": np.array([math.inf] * action_dim),
        },
    }

    metadata_path = Path(info.data_dir) / "metadata.json"
    print(f"Metadata path: {metadata_path}")

    if metadata_path.exists():
        print("Loading dataset statistics from disk...")
        with open(metadata_path, "r") as fd:
            metadata = json.load(fd)
    else:
        for episode in tqdm(ds, desc="Computing dataset statistics"):
            for step in episode["steps"]:
                action = step["action"].numpy()
                metadata["action"]["max"] = np.maximum(
                    metadata["action"]["max"], action
                )
                metadata["action"]["min"] = np.minimum(
                    metadata["action"]["min"], action
                )
        metadata["action"]["max"] = list(metadata["action"]["max"])
        metadata["action"]["min"] = list(metadata["action"]["min"])
        print(f"Saving dataset statistics to disk... {metadata_path}")
        with open(metadata_path, "w") as fd:
            json.dump(metadata, fd)

    return ds, val_ds, metadata


def data_generator(ds_iter, batch_size, chunk_size):
    while True:
        observations = []
        actions = []

        episodes = []
        for _ in range(batch_size):
            episode = next(ds_iter)
            episodes.append(episode)

        for episode in episodes:
            observations.append([])
            actions.append([])
            steps = episode["steps"]
            for step in steps:
                obs = step["observation"]
                action = step["action"]
                observations[-1].append(obs)
                actions[-1].append(action)

        # trim episode data
        observations = [x[4:] for x in observations]
        actions = [x[4:] for x in actions]

        max_length = max(len(a) for a in actions)
        action_shape = actions[0][0].shape
        for b in range(max_length - chunk_size + 1):
            sample_actions = []
            for traj in actions:
                traj = traj[b : b + chunk_size]
                traj_length = len(traj)
                if traj_length < chunk_size:
                    pad_length = chunk_size - traj_length
                    traj += [np.zeros(action_shape)] * pad_length
                sample_actions.append(
                    [(a.numpy() if not isinstance(a, np.ndarray) else a) for a in traj]
                )

            train_sample = {
                "observations": torch.tensor(
                    [x[b if b < len(x) else -1]["image"].numpy() for x in observations]
                ),
                "actions": torch.tensor(sample_actions),
                "frame_index": b,
            }
            yield train_sample


@dataclass
class TrainingConfig:
    # Dataset and Dataloader parameters
    ds_name: str = "mikasa_robo_tfds/ShellGameTouch-v0"
    data_dir: str = "/home/liyouzhou/tensorflow_datasets/"
    batch_size: int = 32
    action_dim: int = 7

    # Model parameters
    model_name: str = "vit_base_patch16_224"

    # Training parameters
    lr: float = 0.0001
    n_steps: int = 10000

    save_steps: int = 5000
    val_steps: int = 1000

    model_config: ViTMemoryConfig = field(default_factory=ViTMemoryConfig)


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=TrainingConfig)


@hydra.main(version_base=None, config_name="config")
def main(cfg: TrainingConfig):
    prevent_tf_gpu_memory_grab()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    random_suffix = "".join(
        secrets.choice(string.ascii_lowercase + string.digits) for i in range(8)
    )
    log_dir = Path("logs") / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + random_suffix
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project="vit-memory", config=cfg_dict, dir=str(log_dir))

    ds, val_ds, metadata = load_dataset(cfg.ds_name, cfg.data_dir, cfg.action_dim)
    print("Dataset statistics:", metadata)

    ds = ds.shuffle(100).repeat().prefetch(cfg.batch_size * 2)  # infinite stream
    ds_iter = iter(ds)

    data_iter = data_generator(ds_iter, cfg.batch_size, cfg.model_config.chunk_size)

    model = DINOv2wMemory(
        cfg.model_config,
        dataset_metadata=metadata,
    )
    model.to("cuda")
    model.train()
    model.freeze_encoder()

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=cfg.lr,
    )

    loss_window = []

    main_pbar = trange(cfg.n_steps)
    for i in main_pbar:
        os.environ["TRAINING_STEP"] = str(i)

        # print("Fetching data...")
        data = next(data_iter)
        # print("Step done.")

        # print("frame_index", data["frame_index"])
        # print(data["actions"].mean())
        # print(data["observations"].float().mean())
        # print("episode_index", data["episode_index"])

        # print(data["frame_index"][0], data["frame_index"][0] == 0)
        if cfg.model_config.enable_memory and data["frame_index"] == 0:
            model.memory.reset_memory()

        # print("data keys:", data.keys())
        # imgs, labels = data

        os.makedirs("image_debug", exist_ok=True)
        if i < 100:
            obs = data["observations"]
            try:
                obs_np = obs.numpy()
            except Exception:
                obs_np = obs.cpu().numpy()
            for b_idx in range(obs_np.shape[0]):
                img = obs_np[b_idx]
                if img.ndim == 3 and img.shape[2] == 1:
                    img = img.squeeze(2)
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                Image.fromarray(img).save(
                    f"image_debug/step_{i:04d}_frame_{data['frame_index']}_b{b_idx:03d}.png"
                )

        imgs = data["observations"].float().to("cuda")

        imgs = rearrange(imgs, "b h w c -> b c h w")
        action = data["actions"].float()

        # print("imgs min/max", imgs.min().item(), imgs.max().item())

        # print(imgs.shape)
        # print(action.shape)
        # print(imgs.device)
        # print(imgs.type)

        # imgs = nn.functional.interpolate(
        #     imgs, size=(224, 224), mode="bilinear", align_corners=False
        # )

        # print("imgs.shape", imgs.shape)

        # print("Forward pass...")
        pred = model(imgs)
        # print("Step done.")

        normalized_action = normalize(
            action,
            torch.tensor(metadata["action"]["min"]),
            torch.tensor(metadata["action"]["max"]),
        )
        normalized_action = normalized_action.to("cuda")

        # print("pred.shape", pred.shape)
        # print("normalized_action.shape", normalized_action.shape)

        pred = pred.view(-1, cfg.model_config.chunk_size, cfg.model_config.action_dim)

        # print("Computing loss...")
        # If GT action is all zeros for a timestep, mask it out from the loss
        # action: (batch, chunk_size, action_dim)
        mask = (action.abs().sum(dim=-1) != 0).to(pred.device)  # (batch, chunk)
        abs_err = torch.abs(pred - normalized_action)  # (batch, chunk, action_dim)
        masked_abs_err = abs_err * mask.unsqueeze(-1).float()

        num_unmasked = mask.sum() * pred.shape[-1]  # scalar tensor
        if num_unmasked.item() > 0:
            loss = masked_abs_err.sum() / num_unmasked
        else:
            # No supervised targets in this batch/step: zero loss (keep requires_grad)
            loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        # print(loss)[]

        # print(
        # torch.cuda.memory_summary(),
        # torch.cuda.max_memory_allocated())
        unnormalized_pred = unnormalize(
            pred.clone().detach().cpu(),
            torch.tensor(metadata["action"]["min"]),
            torch.tensor(metadata["action"]["max"]),
        )
        mse = nn.MSELoss(reduction="mean")(unnormalized_pred, action)
        wandb.log(
            {
                "train/mse": mse.item(),
            },
            step=i,
        )
        # for name, p in model.named_parameters():
        #     print(f"{name:60s} {tuple(p.shape)!s:20s} {p.numel()/1e6:8.2f}M")

        # for action_i, action_val in enumerate(action):
        #     wandb.log(
        #         {
        #             f"train/sample_action/{action_i}": action_val.mean().item(),
        #         },
        #         step=i,
        #     )

        # print("Backward pass...")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print("Step done.")

        # print("pred", pred.argmax(dim=1))
        # print("labels", labels)
        # print("gt", gt)

        # accuracy = (pred.argmax(dim=1) == gt.to("cuda")).float().mean()
        loss_value = loss.item()

        loss_per_dim = abs(normalized_action - pred.detach())

        for j in range(loss_per_dim.shape[2]):
            wandb.log(
                {
                    f"loss/action_dim{j}": loss_per_dim[:, :, j].mean().item(),
                },
                step=i,
            )

        for j in range(loss_per_dim.shape[1]):
            wandb.log(
                {
                    f"loss/action_step{j}": loss_per_dim[:, j, :].mean().item(),
                },
                step=i,
            )

        loss_window.append(loss_value)
        # accuracy_window.append(accuracy.item())
        window_size = 100

        average_loss = sum(loss_window[-window_size:]) / len(loss_window[-window_size:])
        # average_accuracy = sum(accuracy_window[-window_size:]) / len(
        #     accuracy_window[-window_size:]
        # )
        loss_window = loss_window[-window_size:]
        # accuracy_window = accuracy_window[-window_size:]

        wandb.log(
            {
                "train/loss": loss_value,
            },
            step=i,
        )

        wandb.log(
            {
                f"frame/loss/{data['frame_index']}": loss_value,
            },
            step=i,
        )
        main_pbar.set_postfix({f"Loss:": f"{average_loss:.4f}"})

        if (
            cfg.val_steps > 0
            and ((i + 1) % cfg.val_steps == 0)
            or (i + 1) == cfg.n_steps
        ):
            val_ds_iter = iter(val_ds)
            val_data_iter = data_generator(
                val_ds_iter, cfg.batch_size, cfg.model_config.chunk_size
            )

            # Validation
            model.eval()

            val_losses = []
            val_mses = []
            with torch.no_grad():
                for _ in trange(18, desc="Validation", position=1):
                    val_data = next(val_data_iter)
                    if cfg.model_config.enable_memory and val_data["frame_index"] == 0:
                        model.memory.reset_memory()
                    val_imgs = val_data["observations"].float().to("cuda")
                    val_imgs = rearrange(val_imgs, "b h w c -> b c h w")
                    val_action = val_data["actions"].float()

                    val_pred = model(val_imgs)

                    normalized_val_action = normalize(
                        val_action,
                        torch.tensor(metadata["action"]["min"]),
                        torch.tensor(metadata["action"]["max"]),
                    )
                    normalized_val_action = normalized_val_action.to("cuda")

                    val_pred = val_pred.view(
                        -1, cfg.model_config.chunk_size, cfg.model_config.action_dim
                    )

                    val_loss = nn.L1Loss()(val_pred, normalized_val_action)
                    unnormalized_val_pred = unnormalize(
                        val_pred.clone().detach().cpu(),
                        torch.tensor(metadata["action"]["min"]),
                        torch.tensor(metadata["action"]["max"]),
                    )
                    val_mse = nn.MSELoss(reduction="mean")(
                        unnormalized_val_pred, val_action
                    )

                    val_losses.append(val_loss.item())
                    val_mses.append(val_mse.item())

            average_val_loss = sum(val_losses) / len(val_losses)
            average_val_mse = sum(val_mses) / len(val_mses)

            wandb.log(
                {
                    "val/loss": average_val_loss,
                    "val/mse": average_val_mse,
                },
                step=i,
            )
            print(
                f"Validation Loss: {average_val_loss:.4f}, MSE: {average_val_mse:.4f}"
            )
            model.train()
            model.freeze_encoder()
            if cfg.model_config.enable_memory:
                model.memory.reset_memory()

        if cfg.save_steps > 0 and (
            (i + 1) % cfg.save_steps == 0 or (i + 1) == cfg.n_steps
        ):
            save_path = log_dir / f"vit_memory_mikasa_step_{i+1}.safetensors"
            save_file(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")


if __name__ == "__main__":
    main()
