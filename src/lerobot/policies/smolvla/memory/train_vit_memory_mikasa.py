import json
import os
from pathlib import Path
from einops import rearrange
from safetensors.torch import save_file

import hydra
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm, trange

import wandb
from lerobot.policies.smolvla.memory.ViTMemory import (
    EUPEwMemory,
)
from lerobot.policies.smolvla.memory.image_utils import crop_resize
from lerobot.policies.smolvla.memory.training_config import TrainingConfig
from datetime import datetime
import secrets
import string

from lerobot.policies.smolvla.memory.run_mikasa_eval import eval_mikasa
from scipy.spatial.transform import Rotation
from lerobot.policies.smolvla.memory.run_mikasa_eval import (
    GenerateConfig as MikasaEvalConfig,
)


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
            "max": None,
            "min": None,
        },
        "state": {
            "max": None,
            "min": None,
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
                    (
                        metadata["action"]["max"]
                        if metadata["action"]["max"] is not None
                        else action
                    ),
                    action,
                )
                metadata["action"]["min"] = np.minimum(
                    (
                        metadata["action"]["min"]
                        if metadata["action"]["min"] is not None
                        else action
                    ),
                    action,
                )
                state = step["observation"]["state"].numpy()
                metadata["state"]["max"] = np.maximum(
                    (
                        metadata["state"]["max"]
                        if metadata["state"]["max"] is not None
                        else state
                    ),
                    state,
                )
                metadata["state"]["min"] = np.minimum(
                    (
                        metadata["state"]["min"]
                        if metadata["state"]["min"] is not None
                        else state
                    ),
                    state,
                )

        print(f"Saving dataset statistics to disk... {metadata_path}")
        with open(metadata_path, "w") as fd:
            for key in metadata:
                metadata[key]["max"] = metadata[key]["max"].tolist()
                metadata[key]["min"] = metadata[key]["min"].tolist()
            json.dump(metadata, fd)

    return ds, val_ds, metadata


def data_generator(
    ds_iter,
    batch_size,
    chunk_size,
    episode_start_index=0,
    episode_end_index=0,
    downsample_rate=1,
    image_key=["image", "wrist_image"],
    predict_final_pose=False,
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
                action = step["action"].numpy()
                observations[-1].append(obs)
                action[-1] = 0.0  # ignore gripper action
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

            observations_array = np.array(
                [
                    np.concatenate(
                        [x[b if b < len(x) else -1][key].numpy() for key in image_key],
                        axis=-1,
                    )
                    for x in observations
                ]
            )
            embeddings_array = np.array(
                [
                    np.stack(
                        [
                            x[b if b < len(x) else -1][f"{key}_embedding"].numpy()
                            for key in image_key
                        ],
                        axis=0,
                    )
                    for x in observations
                ]
            )
            actions_array = np.array(sample_actions)

            def form_cononical_state(idx):
                # ignore gripper for now
                gripper = np.array([[0] for x in observations])
                state_array = np.array(
                    [
                        x[idx if idx < len(x) else -1]["state"].numpy()
                        for x in observations
                    ]
                )
                quaternion = state_array[:, 3:7]
                euler_angles = Rotation.from_quat(
                    quaternion, scalar_first=True
                ).as_euler("XYZ")
                state_array = np.concatenate(
                    [state_array[:, :3], euler_angles, gripper], axis=1
                )
                state_array = np.expand_dims(state_array, axis=1)
                return state_array

            # calculate the final state of the episode
            final_state_array = form_cononical_state(-1)

            # extract the current state
            current_state_array = form_cononical_state(b).squeeze(1)

            # Extract the next few states as delta to the current state, and use that as the target to predict
            def state_diff(current_state, next_state):
                if current_state.ndim == 3 and current_state.shape[1] == 1:
                    current_state = current_state.squeeze(1)
                if next_state.ndim == 3 and next_state.shape[1] == 1:
                    next_state = next_state.squeeze(1)

                # position difference
                pos_diff = next_state[:, :3] - current_state[:, :3]

                # rotation difference (in euler angles)
                current_rot = Rotation.from_euler("XYZ", current_state[:, 3:6])
                next_rot = Rotation.from_euler("XYZ", next_state[:, 3:6])
                rot_diff = (next_rot * current_rot.inv()).as_euler("XYZ")

                # concatenate position and rotation differences
                gripper = np.array([[0] for _ in observations])
                diff = np.concatenate([pos_diff, rot_diff, gripper], axis=1)

                return diff

            next_states_array = np.array(
                [
                    state_diff(current_state_array, form_cononical_state(b + i))
                    for i in range(chunk_size)
                ]
            )

            train_sample = {
                "observations": torch.from_numpy(observations_array),
                "actions": torch.from_numpy(
                    final_state_array if predict_final_pose else actions_array
                ),
                "frame_index": b,
                "state": torch.from_numpy(current_state_array),
                "embeddings": torch.from_numpy(embeddings_array),
                "next_states": rearrange(
                    torch.from_numpy(next_states_array), "c b d -> b c d"
                ),
            }
            train_episode.append(train_sample)

        yield train_episode


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
    if cfg.predict_relative_states:
        # need to estimate the bounds during training
        metadata["action"]["max"] = np.ones_like(metadata["action"]["max"]) * -np.inf
        metadata["action"]["min"] = np.ones_like(metadata["action"]["min"]) * np.inf
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
        cfg.predict_final_pose,
    )

    # pretend the final state is the "action" to predict if predict_final_pose is True
    if cfg.predict_final_pose:
        metadata["action"]["max"] = metadata["state"]["max"]
        metadata["action"]["min"] = metadata["state"]["min"]

    model = EUPEwMemory(
        cfg.model_config,
        dataset_metadata=metadata,
    )
    model.to("cuda")
    model.train()
    model.freeze_encoder()
    transform = T.RandomResizedCrop(size=128, scale=(0.9, 0.9), ratio=(1, 1))

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

        # pick a random crop for the batch, and apply the same crop to all frames in that batch
        offset_limit = int(
            data[0]["observations"].shape[1]
            * (1 - cfg.image_augmentation_crop_factor)
            / 2
        )
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
                crop_offset_x = np.random.randint(-offset_limit, offset_limit)
                crop_offset_y = np.random.randint(-offset_limit, offset_limit)
                imgs = crop_resize(
                    imgs,
                    factor=cfg.image_augmentation_crop_factor,
                    crop_offset_x=crop_offset_x,
                    crop_offset_y=crop_offset_y,
                )

            state = None
            if cfg.model_config.proprioception:
                state = data[frame_idx]["state"].float().to("cuda")

            embaddings = data[frame_idx]["embeddings"].float().to("cuda")
            embeddings = rearrange(embaddings, "b n w h -> (b n) w h")

            if cfg.predict_relative_states:
                action = data[frame_idx]["next_states"].float().to("cuda")
                rearranged_action = rearrange(
                    action[:, :, : cfg.model_config.action_dim], "b c d -> (b c) d"
                )
                model.action_min = torch.min(
                    model.action_min, rearranged_action.min(dim=0).values
                )
                model.action_max = torch.max(
                    model.action_max, rearranged_action.max(dim=0).values
                )

            pred, out_features = model(imgs, state, features=embeddings)
            loss_for_this_frame = model.compute_loss(pred, action)

            if data[frame_idx]["frame_index"] < cfg.action_start_index:
                # No supervised targets in this batch/step: zero loss (keep requires_grad)
                loss += torch.tensor(0.0, device=pred.device)
            else:
                loss += loss_for_this_frame

            loss_value = loss.item()
            if cfg.backprop_every_frame:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = torch.tensor(0.0, device=pred.device)

            mse = model.compute_mse(pred, action)
            mse_values.append(mse.item())

            normalized_action = model.normalize(
                action[:, :, : cfg.model_config.action_dim],
            )
            pred = pred.detach().view(
                -1, cfg.model_config.chunk_size, cfg.model_config.action_dim
            )
            pred = pred.to(normalized_action.device)
            loss_per_dim_values.append(abs(normalized_action - pred.detach()))
            episode_loss.append(loss_value)

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
                cfg.predict_final_pose,
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
                        if cfg.image_augmentation:
                            val_imgs = crop_resize(
                                val_imgs,
                                factor=cfg.image_augmentation_crop_factor,
                            )
                        val_action = val_data[frame_idx]["actions"].float()

                        state = (
                            val_data[frame_idx]["state"].float().to("cuda")
                            if "state" in val_data[frame_idx]
                            else None
                        )

                        val_pred, _ = model(val_imgs, state=state)
                        if cfg.predict_relative_states:
                            val_action = (
                                val_data[frame_idx]["next_states"].float().to("cuda")
                            )
                            val_loss = model.compute_loss(
                                val_pred, val_action, normalize_actions=False
                            )
                        else:
                            val_loss = model.compute_loss(val_pred, val_action)
                        val_mse = model.compute_mse(
                            val_pred,
                            val_action,
                            unnormalize_actions=(not cfg.predict_relative_states),
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

        if cfg.eval_steps > 0 and (
            (i + 1) % cfg.eval_steps == 0 or (i + 1) == cfg.n_steps
        ):
            # Evaluation code can be added here, e.g., visualizing predictions vs GT
            eval_cfg = MikasaEvalConfig(
                task_suite_name=cfg.task_suite_name,
                model_action_scale=cfg.model_action_scale,
                num_envs=cfg.num_envs,
                num_trials_per_task=cfg.num_trials_per_task,
                use_wandb=True,
                log_performance_graphs=False,
                log_rollout_videos=True,  # only log videos when saving checkpoints
                reset_action_cache_every_step=cfg.reset_action_cache_every_step,
                center_crop_images=cfg.image_augmentation,
                crop_factor=cfg.image_augmentation_crop_factor,
                final_pose_as_target=cfg.predict_final_pose,
                predict_relative_states=cfg.predict_relative_states,
            )
            eval_mikasa(
                cfg=eval_cfg,
                model=model,
                skip_wandb_init=True,
                training_step=i,
            )


if __name__ == "__main__":
    main()
