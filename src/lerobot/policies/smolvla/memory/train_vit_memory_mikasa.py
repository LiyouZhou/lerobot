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
from torchvision import transforms as T
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

    available_splits = list(info.splits.keys())

    val_split_name = "train"
    if "val" in available_splits:
        val_split_name = "val"
    elif "test" in available_splits:
        val_split_name = "test"

    val_ds = tfds.load(
        ds_name,
        split=val_split_name,
        data_dir=data_dir,
        download=False,
    )

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


def data_generator(
    ds_iter,
    batch_size,
    chunk_size,
    episode_start_index=0,
    episode_end_index=0,
    downsample_rate=1,
    image_key="image",
):
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

        def pad_to_chunk_size(traj, chunk_size):
            traj_length = len(traj)
            pad_length = max(chunk_size - traj_length, 0)
            pad_value = np.zeros_like(traj[-1])
            traj += [pad_value] * pad_length
            return traj[:chunk_size]

        chunked_actions = [
            [pad_to_chunk_size(traj[i:], chunk_size) for i in range(len(traj))]
            for traj in actions
        ]

        # trim episode data
        observations = [x[episode_start_index:] for x in observations]
        chunked_actions = [x[episode_start_index:] for x in chunked_actions]

        # Downsample
        observations = [x[::downsample_rate] for x in observations]
        chunked_actions = [x[::downsample_rate] for x in chunked_actions]

        max_length = max(len(a) for a in chunked_actions)
        action_shape = chunked_actions[0][0][0].shape

        num_frames_in_episode = min(
            (max_length - chunk_size + 1) if max_length >= chunk_size else max_length,
            (
                (episode_end_index - episode_start_index)
                if episode_end_index > episode_start_index
                else max_length
            ),
        )
        train_episode = []
        for b in range(num_frames_in_episode):
            sample_actions = []
            for traj in chunked_actions:
                if b >= len(traj):
                    chunk = [np.zeros(action_shape) for _ in range(chunk_size)]
                else:
                    chunk = traj[b]
                sample_actions.append(
                    [(a.numpy() if not isinstance(a, np.ndarray) else a) for a in chunk]
                )

            observations_array = np.array([
                x[b if b < len(x) else -1][image_key].numpy()
                for x in observations
            ])
            actions_array = np.array(sample_actions)
            train_sample = {
                "observations": torch.from_numpy(observations_array),
                "actions": torch.from_numpy(actions_array),
                "frame_index": b,
            }
            train_episode.append(train_sample)

        yield train_episode


@dataclass
class TrainingConfig:
    # Dataset and Dataloader parameters
    ds_name: str = "mikasa_robo_tfds/ShellGameTouch-v0"
    data_dir: str = "/home/liyouzhou/tensorflow_datasets/"
    image_key: str = "image"
    batch_size: int = 32
    episode_start_index: int = 0
    episode_end_index: int = 0  # 0 means till the end
    downsample_rate: int = 1
    # action only applies after this index in each episode this is counting after trim and downsample
    action_start_index: int = 0

    # Model parameters
    model_name: str = "dino_v2-base"

    # Training parameters
    lr: float = 0.0001
    n_steps: int = 10000

    save_steps: int = 5000
    val_steps: int = 1000
    num_val_steps: int = 20

    model_config: ViTMemoryConfig = field(default_factory=ViTMemoryConfig)

    image_debug: bool = False
    image_augmentation: bool = False
    backprop_every_frame: bool = True


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
    with open(log_dir / "config.yaml", "w") as fd:
        fd.write(OmegaConf.to_yaml(cfg))
    print("Config", cfg_dict)
    print(f"Logging to directory: {log_dir}")

    wandb.init(project="vit-memory", config=cfg_dict, dir=str(log_dir))

    ds, val_ds, metadata = load_dataset(
        cfg.ds_name, cfg.data_dir, cfg.model_config.action_dim
    )
    print("Dataset statistics:", metadata)

    val_ds = val_ds.shuffle(50).repeat().prefetch(cfg.batch_size * 2)  # infinite stream
    ds = ds.shuffle(100).repeat().prefetch(cfg.batch_size * 2)  # infinite stream
    ds_iter = iter(ds)

    data_iter = data_generator(
        ds_iter,
        cfg.batch_size,
        cfg.model_config.chunk_size,
        cfg.episode_start_index,
        cfg.episode_end_index,
        cfg.downsample_rate,
        cfg.image_key,
    )

    model = DINOv2wMemory(
        cfg.model_config,
        dataset_metadata=metadata,
    )
    model.to("cuda")
    model.train()
    model.freeze_encoder()
    transform = T.RandomResizedCrop(size=128, scale=(0.8, 1.0), ratio=(1, 1))

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=cfg.lr,
    )

    loss_window = []
    main_pbar = trange(cfg.n_steps)
    for i in main_pbar:
        os.environ["TRAINING_STEP"] = str(i)

        # sample an batch of episodes
        data = next(data_iter)

        # reset memory at the start of each episode
        model.reset_memory()

        loss = torch.tensor(0.0, device="cuda")
        episode_loss = []
        mse_values = []
        loss_per_dim_values = []
        for frame_idx in range(len(data)):
            if cfg.image_debug and i < 100:
                os.makedirs("image_debug", exist_ok=True)
                obs = data[frame_idx]["observations"]
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
                        f"image_debug/step_{i:04d}_frame_{data[frame_idx]['frame_index']}_b{b_idx:03d}.png"
                    )

            imgs = data[frame_idx]["observations"].float().to("cuda")
            action = data[frame_idx]["actions"].float()

            if cfg.image_augmentation:
                imgs = rearrange(imgs, "b h w c -> b c h w")
                imgs = torch.stack([transform(img) for img in imgs])

            pred, out_features = model(imgs)

            normalized_action = normalize(
                action[:, :, : cfg.model_config.action_dim],
                torch.tensor(metadata["action"]["min"]),
                torch.tensor(metadata["action"]["max"]),
            )
            normalized_action = normalized_action.to("cuda")

            pred = pred.view(
                -1, cfg.model_config.chunk_size, cfg.model_config.action_dim
            )

            # If GT action is all zeros for a timestep, mask it out from the loss
            # action: (batch, chunk_size, action_dim)
            mask = (action.abs().sum(dim=-1) != 0).to(pred.device)  # (batch, chunk)
            abs_err = torch.abs(pred - normalized_action)  # (batch, chunk, action_dim)
            masked_abs_err = abs_err * mask.unsqueeze(-1).float()

            num_unmasked = mask.sum() * pred.shape[-1]  # scalar tensor
            if (
                data[frame_idx]["frame_index"] < cfg.action_start_index
                or num_unmasked.item() == 0
            ):
                # No supervised targets in this batch/step: zero loss (keep requires_grad)
                loss += torch.tensor(0.0, device=pred.device)
            else:
                loss += masked_abs_err.sum() / num_unmasked

            if cfg.backprop_every_frame:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = torch.tensor(0.0, device=pred.device)

            unnormalized_pred = unnormalize(
                pred.clone().detach().cpu(),
                torch.tensor(metadata["action"]["min"]),
                torch.tensor(metadata["action"]["max"]),
            )
            mse = nn.MSELoss(reduction="mean")(
                unnormalized_pred, action[:, :, : cfg.model_config.action_dim]
            )
            mse_values.append(mse.item())
            loss_per_dim_values.append(abs(normalized_action - pred.detach()))
            episode_loss.append(loss.item())

        if not cfg.backprop_every_frame:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_value = loss.item()
        # average over episode and batch
        mean_loss_per_dim = (
            torch.stack(loss_per_dim_values).mean(dim=0).mean(dim=0)
        )  # (chunk_size, action_dim)
        wandb.log(
            {
                "train/loss": loss_value,
                f"train/mse": (
                    sum(mse_values) / len(mse_values) if len(mse_values) > 0 else 0.0
                ),
                "train/episode_loss": (
                    sum(episode_loss) / len(episode_loss)
                    if len(episode_loss) > 0
                    else 0.0
                ),
                "train/episode_length": len(episode_loss),
                **{
                    f"loss/action_dim{j}": mean_loss_per_dim[:, j].mean().item()
                    for j in range(mean_loss_per_dim.shape[1])
                },
                **{
                    f"loss/action_step{j}": mean_loss_per_dim[j, :].mean().item()
                    for j in range(mean_loss_per_dim.shape[0])
                },
                **{f"frame/loss/{j:03}": l for j, l in enumerate(episode_loss)},
            },
            step=i,
        )

        window_size = 100
        loss_window.append(loss_value)
        average_loss = sum(loss_window[-window_size:]) / len(loss_window[-window_size:])
        loss_window = loss_window[-window_size:]
        main_pbar.set_postfix({f"Loss:": f"{average_loss:.4f}"})

        if (
            cfg.val_steps > 0
            and ((i + 1) % cfg.val_steps == 0)
            or (i + 1) == cfg.n_steps
        ):
            val_ds_iter = iter(val_ds)
            val_data_iter = data_generator(
                val_ds_iter,
                cfg.batch_size,
                cfg.model_config.chunk_size,
                cfg.episode_start_index,
                cfg.episode_end_index,
                cfg.downsample_rate,
                cfg.image_key,
            )

            # Validation
            model.eval()

            val_losses = []
            val_mses = []
            with torch.no_grad():
                for _ in trange(cfg.num_val_steps, desc="Validation", position=1):
                    # sample a batch of episodes
                    val_data = next(val_data_iter)

                    model.reset_memory()

                    for frame_idx in range(len(val_data)):
                        val_imgs = (
                            val_data[frame_idx]["observations"].float().to("cuda")
                        )
                        val_action = val_data[frame_idx]["actions"].float()
                        val_action = val_action[:, :, : cfg.model_config.action_dim]

                        val_pred, _ = model(val_imgs)

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
            model.reset_memory()

        if cfg.save_steps > 0 and (
            (i + 1) % cfg.save_steps == 0 or (i + 1) == cfg.n_steps
        ):
            save_path = log_dir / f"vit_memory_mikasa_step_{i+1}.safetensors"
            save_file(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")


if __name__ == "__main__":
    main()
