#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
from pathlib import Path

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler, EpisodicBatchSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.scripts.run_mikasa_eval import eval_mikasa, GenerateConfig
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger
from tqdm import trange
import os
import threading
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import load_smolvla


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{random.randint(12000, 20000)}"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy | DDP,
    dl_iter: Any,
    device: torch.device,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
    device_type: str = "cuda",
    num_accumulation_steps: int = 1,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    # device = get_device_from_parameters(policy)
    policy.train()
    with (
        torch.autocast(device_type=device_type, dtype=torch.float16)
        if use_amp
        else nullcontext()
    ):
        loss_accumulated = 0
        for i in range(num_accumulation_steps):
            start_time = time.perf_counter()
            batch = next(dl_iter)
            train_metrics.dataloading_s = time.perf_counter() - start_time

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(
                        device, non_blocking=device_type == "cuda"
                    )

            # assert batch["frame_index"].shape[0] == 1, "Batch size must be 1"

            if batch["frame_index"][0].cpu().tolist() == 0:
                print("Frame 0, Resetting memory")
                policy.module.model.vlm_with_expert.reset_memory()
            loss, output_dict = policy(batch)
            loss_accumulated += loss
            # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    grad_scaler.scale(loss_accumulated).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time

    ground_truth_actions = output_dict["ground_truth_actions"]
    predicted_actions = output_dict["predicted_actions"]
    frame_indices = batch["frame_index"]
    task_indices = batch["task_index"]

    # Calculate and log per-frame and per-task MSE
    mse = F.mse_loss(predicted_actions, ground_truth_actions, reduction="none")
    mse_per_sample = mse.mean(dim=1)  # Mean over action dimensions
    mse_per_sample = mse_per_sample.mean(dim=1)  # Mean over action dimensions

    current_training_step = int(os.environ.get("CURRENT_TRAINING_STEP", 0))
    for idx, task_idx in enumerate(task_indices.tolist()):
        task_loss = mse_per_sample[idx].item()

        wandb.log(
            {
                f"task_mse/task_{task_idx}": task_loss,
                f"task_mse/training_step": current_training_step,

            }
        )

    wandb.log(
        {
            f"frame_mse/frame_{frame_indices[0].cpu().tolist()}": loss_accumulated.item(),
            f"frame_mse/training_step": current_training_step,
        }
    )

    return train_metrics, output_dict


def train(rank: int, cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if rank == 0 and cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    device_type = device.type

    if torch.cuda.is_available() and device_type == "cuda":
        torch.cuda.set_device(rank)
        device = rank
        device_type = "cuda"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")

    dataset_result = {}

    def load_dataset():
        dataset_result["dataset"] = make_dataset(cfg)

    dataset_thread = threading.Thread(target=load_dataset)
    dataset_thread.start()

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(
            cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
        )

    logging.info(f"Creating policy: {cfg.policy}")

    ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root)
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device_type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())

    dataset_thread.join()
    dataset = dataset_result["dataset"]
    logging.info(
        colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}"
    )
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    if cfg.episodic:
        logging.info(f"Training Episodically")
        batch_sampler = EpisodicBatchSampler(
            repo_root=cfg.dataset.root,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            remember_color_only=cfg.remember_color_only
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_sampler=batch_sampler,
            pin_memory=device_type == "cuda",
        )
    else:
        sampler = DistributedSampler(dataset)
        shuffle = False
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=device_type == "cuda",
            drop_last=False,
        )

    dl_iter = cycle(dataloader)

    policy = DDP(policy, device_ids=[device])
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    logging.info("Start offline training on a fixed dataset")
    is_first_step = True
    for _ in trange(step, cfg.steps, position=rank, desc=f"Rank {rank}"):

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            dl_iter,
            device,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
            device_type=device_type,
        )

        if is_first_step:
            is_first_step = False
            logging.info(
                "First step completed which means memory has finished initialization. Now load mem initialisation weights."
            )
            print(f"Pretrained path: {cfg.policy.pretrained_path}")
            if (
                cfg.policy.pretrained_path is not None
                and Path(cfg.policy.pretrained_path).exists()
            ):
                fn = list(Path(cfg.policy.pretrained_path).glob("*.safetensors"))[
                    0
                ].as_posix()
                load_smolvla(policy.module, fn, device=device)

                policy.module.model.vlm_with_expert.reset_memory()

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        os.environ["CURRENT_TRAINING_STEP"] = str(step)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if rank == 0 and is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)

                wandb_log_dict.update({"train_steps": step})
                wandb_logger.log_dict(wandb_log_dict, custom_step_key="train_steps")
            train_tracker.reset_averages()

        if rank == 0 and cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir, step, cfg, policy.module, optimizer, lr_scheduler
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if rank == 0 and is_eval_step:
            logging.info(f"Eval policy at step {step}")
            eval_cfg = GenerateConfig(
                task_suite_name="mikasa_remember_color",
                num_envs=cfg.num_envs,
                num_trials_per_task=cfg.num_trials_per_task,
                use_wandb=True,
                repo_path=cfg.dataset.root,
                log_performance_graphs=False,
                log_rollout_videos=False,
            )
            eval_mikasa(cfg=eval_cfg, model=policy.module, skip_wandb_init=True)

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


def launch_train_ddp(rank, world_size, cfg):
    print(f"my rank is {rank} / {world_size}")

    try:
        ddp_setup(rank, world_size)
        init_logging()
        train(rank, cfg)
    except Exception as e:
        logging.error(f"Error occurred in DDP process {rank}: {e}")
        raise e
    finally:
        destroy_process_group()


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    world_size = torch.cuda.device_count()
    mp.spawn(launch_train_ddp, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
