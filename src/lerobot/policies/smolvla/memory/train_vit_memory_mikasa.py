from itertools import cycle
import math
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.sampler import EpisodicBatchSampler

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import wandb
import os

from lerobot.policies.smolvla.memory.ViTMemory import DINOv2wMemory, ViTwMemory
from torch import nn
from tqdm import tqdm, trange

from lerobot.utils.utils import print_cuda_memory_usage
import tensorflow as tf
import tensorflow_datasets as tfds
from einops import rearrange
import numpy as np
from pathlib import Path
import json
from PIL import Image


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


if __name__ == "__main__":
    repo_id = "mikasa_robo_tfds_all_1.0.0_lerobot"
    repo_root = "/home/liyouzhou/lerobot_datasets/mikasa_robo_tfds_all_1.0.0_lerobot"
    batch_size = 32
    lr = 0.0001
    episode_length = 5
    enable_memory = False
    inner_lr = 0.01
    chunk_size = 10
    num_workers = 4
    action_dim = 7
    num_classes = action_dim * chunk_size

    # ds_meta = LeRobotDatasetMetadata(repo_id, root=repo_root)
    # action_dim = ds_meta.features["action"]['shape'][-1]

    # dataset = LeRobotDataset(
    #     repo_id=repo_id,
    #     root=repo_root,
    #     delta_timestamps={
    #         "action": [i / ds_meta.fps for i in range(chunk_size)],
    #     },
    #     video_backend="torchcodec"
    # )

    # batch_sampler = EpisodicBatchSampler(
    #     repo_root=repo_root,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     allowable_task_names=["RememberColor3-v0"],
    # )

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     num_workers=num_workers,
    #     batch_sampler=batch_sampler,
    #     pin_memory=False,
    #     persistent_workers=False
    # )
    # dataloader_iter = cycle(dataloader)

    # ds_name = "mikasa_robo_tfds/RememberColor3-v0_baseline"
    ds_name = "mikasa_robo_tfds/ShellGameTouch-v0"
    data_dir = os.path.expanduser("/home/liyouzhou/tensorflow_datasets/")

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    ds, info = tfds.load(
        ds_name, split="train", data_dir=data_dir, with_info=True, download=False
    )

    metadata = {
        "action": {
            "max": np.array([-math.inf] * action_dim),
            "min": np.array([math.inf] * action_dim),
        },
    }

    metadata_path = Path(data_dir) / ds_name / "metadata.json"

    if metadata_path.exists():
        print("Loading dataset statistics from disk...")
        with open(metadata_path, "r") as fd:
            metadata = json.load(fd)
    else:
        for episode in tqdm(ds, desc="Computing dataset statistics"):
            for step in episode["steps"]:
                action = step["action"].numpy()
                metadata["action"]["max"] = np.maximum(metadata["action"]["max"], action)
                metadata["action"]["min"] = np.minimum(metadata["action"]["min"], action)
        metadata["action"]["max"] = list(metadata["action"]["max"])
        metadata["action"]["min"] = list(metadata["action"]["min"])
        print("Saving dataset statistics to disk...")
        with open(metadata_path, "w") as fd:
            json.dump(metadata, fd)

    print("Dataset statistics:", metadata)

    ds = ds.shuffle(100).repeat().prefetch(batch_size*2)  # infinite stream
    ds_iter = iter(ds)

    def data_generator():
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

            for b in range(3):
                sample_actions = []
                for traj in actions:
                    traj = traj[b : b + chunk_size]
                    traj_length = len(traj)
                    if traj_length < chunk_size:
                        pad_length = chunk_size - traj_length
                        traj += [traj[-1]] * pad_length
                    traj = traj[:chunk_size]
                    sample_actions.append([a.numpy() for a in traj])
                # for b in range(batch_size):
                #     observations[b] = observations[b][4:] # Remove first few frames
                #     actions[b] = actions[b][4:]
                #     episode_length = len(observations[b])

                #     if episode_length < chunk_size:
                #         pad_length = chunk_size - episode_length
                #         observations[b] += [observations[b][-1]] * pad_length
                #         actions[b] += [actions[b][-1]] * pad_length

                #     observations[b] = observations[b][:chunk_size]
                #     actions[b] = actions[b][:chunk_size]

                train_sample = {
                    "observations": torch.tensor(
                        [x[b]["image"].numpy() for x in observations]
                    ),
                    "actions": torch.tensor(
                        sample_actions
                    ),
                    "frame_index": b,
                }
                yield train_sample

    data_iter = data_generator()

    model = DINOv2wMemory(
        enable_memory=enable_memory, num_classes=num_classes, inner_lr=inner_lr
    )
    model.to("cuda")
    model.train()
    model.freeze_encoder()

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=lr,
    )

    n_steps = 10000

    loss_window = []
    accuracy_window = []

    config = {
        "lr": lr,
        "batch_size": batch_size,
        "episode_length": episode_length,
        "model_name": "vit_base_patch16_224",
        "enable_memory": enable_memory,
        "inner_lr": inner_lr,
        "num_classes": num_classes,
    }
    wandb.init(project="vit-memory", config=config)

    main_pbar = trange(n_steps)
    for i in main_pbar:
        os.environ["TRAINING_STEP"] = str(i)

        gt = []
        # print("Fetching data...")
        data = next(data_iter)
        # print("Step done.")

        # print("frame_index", data["frame_index"])
        # print(data["actions"].mean())
        # print(data["observations"].float().mean())
        # print("episode_index", data["episode_index"])

        # print(data["frame_index"][0], data["frame_index"][0] == 0)
        if enable_memory and data["frame_index"] == 0:
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
                Image.fromarray(img).save(f"image_debug/step_{i:04d}_frame_{data['frame_index']}_b{b_idx:03d}.png")

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

        pred = pred.view(-1, chunk_size, action_dim)

        # print("Computing loss...")
        # print(pred.shape, action.shape)
        loss = nn.L1Loss()(pred, normalized_action)
        # print(loss)

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

