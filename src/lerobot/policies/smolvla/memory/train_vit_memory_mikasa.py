from itertools import cycle
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
from tqdm import trange

from lerobot.utils.utils import print_cuda_memory_usage


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


if __name__ == "__main__":
    repo_id = "mikasa_robo_tfds_all_1.0.0_lerobot"
    repo_root = "/home/liyouzhou/lerobot_datasets/mikasa_robo_tfds_all_1.0.0_lerobot"
    batch_size = 16
    lr = 0.00001
    batch_size = 32
    episode_length = 5
    enable_memory = False
    inner_lr = 0.01
    chunk_size = 20
    num_workers = 4

    ds_meta = LeRobotDatasetMetadata(repo_id, root=repo_root)
    action_dim = ds_meta.features["action"]['shape'][-1]
    num_classes = action_dim * chunk_size

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=repo_root,
        delta_timestamps={
            "action": [i / ds_meta.fps for i in range(chunk_size)],
        },
        video_backend="torchcodec"
    )

    batch_sampler = EpisodicBatchSampler(
        repo_root=repo_root,
        batch_size=batch_size,
        shuffle=False,
        allowable_task_names=["RememberColor3-v0"],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        pin_memory=False,
        persistent_workers=False
    )
    dataloader_iter = cycle(dataloader)

    model = DINOv2wMemory(
        enable_memory=enable_memory, num_classes=num_classes, inner_lr=inner_lr
    )
    model.to("cuda")
    model.train()
    model.encoder.eval()

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
        print("Fetching data...")
        data = next(dataloader_iter)
        print("Step done.")

        print("frame_index", data["frame_index"])
        print("episode_index", data["episode_index"])

        print(data["frame_index"][0], data["frame_index"][0] == 0)
        if data["frame_index"][0] > 10:
            continue
        if enable_memory and data["frame_index"][0] == 0:
            model.memory.reset_memory()
        # print("data keys:", data.keys())
        # imgs, labels = data

        imgs = data["observation.images.image"].clone()
        action = data["action"]

        print("imgs min/max", imgs.min().item(), imgs.max().item())

        print(imgs.shape)
        print(imgs.device)
        print(imgs.type)
        print_cuda_memory_usage()
        

        # imgs = nn.functional.interpolate(
        #     imgs, size=(224, 224), mode="bilinear", align_corners=False
        # )

        # print("imgs.shape", imgs.shape)

        print("Forward pass...")
        pred = model(imgs)
        print("Step done.")

        normalized_action = normalize(
            action,
            ds_meta.stats["action"]["min"],
            ds_meta.stats["action"]["max"],
        )
        normalized_action = normalized_action.to("cuda")


        # print("pred.shape", pred.shape)
        # print("normalized_action.shape", normalized_action.shape)

        pred = pred.view(-1, chunk_size, action_dim)

        loss = nn.L1Loss()(pred, normalized_action)

        unnormalized_pred = unnormalize(
            pred.clone().detach().cpu(),
            ds_meta.stats["action"]["min"],
            ds_meta.stats["action"]["max"],
        )
        mse = nn.MSELoss(reduction="mean")(unnormalized_pred, action)
        wandb.log(
            {
                "train/mse": mse.item(),
            },
            step=i,
        )

        for action_i, action_val in enumerate(action):
            wandb.log(
                {
                    f"train/sample_action/{action_i}": action_val.mean().item(),
                },
                step=i,
            )

        print("Backward pass...")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Step done.")

        # print("pred", pred.argmax(dim=1))
        # print("labels", labels)
        # print("gt", gt)

        # accuracy = (pred.argmax(dim=1) == gt.to("cuda")).float().mean()
        loss_value = loss.item()

        loss_per_dim = abs(action - pred.detach().cpu())

        for j in range(action.shape[1]):
            wandb.log(
                {
                    f"train/loss/{j}": loss_per_dim[:, j].mean().item(),
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
        main_pbar.set_postfix({f"Loss:": f"{average_loss:.4f}"})
