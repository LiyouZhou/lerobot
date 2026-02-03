from collections import defaultdict
from datetime import datetime
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from xml.parsers.expat import model
import draccus
import einops
import torch

from omegaconf import OmegaConf
import numpy as np
import tqdm

import wandb
import gymnasium as gym
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.memory.ViTMemory import DINOv2wMemory, ViTMemoryConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import load_smolvla
from lerobot.policies.smolvla.memory.train_vit_memory_mikasa import TrainingConfig

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")

import tensorflow as tf
from PIL import Image
import imageio
import numpy as np
import plotly.graph_objects as go  # or: import plotly.graph_objs as go

DATE = datetime.now().strftime("%Y-%m-%d")
DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

from torchvision.io import write_video


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    rollout_images = torch.stack(rollout_images)
    # print("rollout_images", rollout_images.shape)
    write_video(mp4_path, rollout_images, fps=30)
    # print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")

    return mp4_path


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    model_action_scale: float = 10.0                 # Scale for model output actions

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "mikasa"                  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 6                          # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    num_envs: int = 4                                # Number of environments to run in parallel (for multi-agent tasks)

    log_rollout_videos: bool = True                 # Whether to save rollout videos
    log_performance_graphs: bool = True              # Whether to log model graphs to W&B
    #################################################################################################################
    # fmt: on


# fmt: off
PROMPT_TEMPLATE_SHELL_GAME_TOUCH = "Memorize the position of the ball, then touch the cup with ball."
PROMPT_TEMPLATE_SHELL_GAME_PUSH = "Memorize the position of the ball, then push the cup with ball."
PROMPT_TEMPLATE_SHELL_GAME_PICK = "Memorize the position of the ball, then pick up the cup with ball."
PROMPT_TEMPLATE_INTERCEPT = "Intercept the rolling ball and guide it towards the target."
PROMPT_TEMPLATE_INTERCEPT_GRAB = "Intercept the rolling ball. Then catch the ball with the gripper and lift it up."
PROMPT_TEMPLATE_ROTATE_LENIENT = "Memorize the initial position of the peg and rotate it back to its initial position."
PROMPT_TEMPLATE_ROTATE_STRICT = "Memorize the initial position of the peg and rotate it back to its initial position without shifting its center."
PROMPT_TEMPLATE_TAKE_IT_BACK = "Memorize the initial position of the cube, move it to the target region, and then return it to its initial position."
PROMPT_TEMPLATE_REMEMBER_COLOR = "Memorize the the colors of the cube shown on the table, and then touch the same coloured cube out of all the cubes."
PROMPT_TEMPLATE_REMEMBER_SHAPE = "Memorize the the shapes of the block shown on the table, and then touch the same shaped blocks."
PROMPT_TEMPLATE_REMEMBER_SHAPE_AND_COLOR = "Memorize the shape and color of the blocks shown, and touch the blocks with the same shape and color."
PROMPT_TEMPLATE_BUNCH_OF_COLORS = "Remember colors of the blocks shown at the begining, touch the same colored blocks in any order."
PROMPT_TEMPLATE_SEQ_OF_COLORS = "Remember the colors of the set of cubes shown sequentially and then touch them in any order."
PROMPT_TEMPLATE_CHAIN_OF_COLORS = "Remember the colors of the set of cubes shown sequentially and then select them in the same order as shown."
# fmt: on

TASK_PROMPTS = {
    "ShellGameTouch-v0": PROMPT_TEMPLATE_SHELL_GAME_TOUCH,
    "ShellGamePush-v0": PROMPT_TEMPLATE_SHELL_GAME_PUSH,
    "ShellGamePick-v0": PROMPT_TEMPLATE_SHELL_GAME_PICK,
    "InterceptSlow-v0": PROMPT_TEMPLATE_INTERCEPT,
    "InterceptMedium-v0": PROMPT_TEMPLATE_INTERCEPT,
    "InterceptFast-v0": PROMPT_TEMPLATE_INTERCEPT,
    "InterceptGrabSlow-v0": PROMPT_TEMPLATE_INTERCEPT_GRAB,
    "InterceptGrabMedium-v0": PROMPT_TEMPLATE_INTERCEPT_GRAB,
    "InterceptGrabFast-v0": PROMPT_TEMPLATE_INTERCEPT_GRAB,
    "RotateLenientPos-v0": PROMPT_TEMPLATE_ROTATE_LENIENT,
    "RotateLenientPosNeg-v0": PROMPT_TEMPLATE_ROTATE_LENIENT,
    "RotateStrictPos-v0": PROMPT_TEMPLATE_ROTATE_STRICT,
    "RotateStrictPosNeg-v0": PROMPT_TEMPLATE_ROTATE_STRICT,
    "TakeItBack-v0": PROMPT_TEMPLATE_TAKE_IT_BACK,
    "RememberColor3-v0": PROMPT_TEMPLATE_REMEMBER_COLOR,
    "RememberColor5-v0": PROMPT_TEMPLATE_REMEMBER_COLOR,
    "RememberColor9-v0": PROMPT_TEMPLATE_REMEMBER_COLOR,
    "RememberShape3-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "RememberShape5-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "RememberShape9-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "RememberShapeAndColor3x2-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "RememberShapeAndColor3x3-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "RememberShapeAndColor5x3-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "BunchOfColors3-v0": PROMPT_TEMPLATE_BUNCH_OF_COLORS,
    "BunchOfColors5-v0": PROMPT_TEMPLATE_BUNCH_OF_COLORS,
    "BunchOfColors7-v0": PROMPT_TEMPLATE_BUNCH_OF_COLORS,
    "SeqOfColors3-v0": PROMPT_TEMPLATE_SEQ_OF_COLORS,
    "SeqOfColors5-v0": PROMPT_TEMPLATE_SEQ_OF_COLORS,
    "SeqOfColors7-v0": PROMPT_TEMPLATE_SEQ_OF_COLORS,
    "ChainOfColors3-v0": PROMPT_TEMPLATE_CHAIN_OF_COLORS,
    "ChainOfColors5-v0": PROMPT_TEMPLATE_CHAIN_OF_COLORS,
    "ChainOfColors7-v0": PROMPT_TEMPLATE_CHAIN_OF_COLORS,
}

TEST_SUITES = {
    "mikasa": {
        "tasks": [
            {
                "task_name": "RememberColor3-v0_baseline",
                "env_name": "RememberColor3-v0",
                "baseline_prompt": True,
                "prompt": "Touch the red cube.",
            },
            {
                "task_name": "RememberColor9-v0_baseline",
                "env_name": "RememberColor9-v0",
                "baseline_prompt": True,
                "prompt": "Touch the red cube.",
            },
        ]
    }
}

for task_name, prompt in TASK_PROMPTS.items():
    TEST_SUITES["mikasa"]["tasks"].append(
        {
            "task_name": task_name,
            "env_name": task_name,
            "baseline_prompt": False,
            "prompt": prompt,
        }
    )

TEST_SUITES["mikasa_remember_color"] = {
    "tasks": [
        task
        for task in TEST_SUITES["mikasa"]["tasks"]
        if "RememberColor" in task["task_name"]
    ]
}

TEST_SUITES["mikasa_baseline"] = {
    "tasks": [
        task
        for task in TEST_SUITES["mikasa"]["tasks"]
        if "baseline" in task["task_name"]
    ]
}

TEST_SUITES["mikasa_subset"] = {
    "tasks": [
        task
        for task in TEST_SUITES["mikasa"]["tasks"]
        if (
            ("RememberColor" in task["task_name"])
            and ("baseline" not in task["task_name"])
        )
        or (task["task_name"] in ["ShellGameTouch-v0", "InterceptMedium-v0"])
    ]
}

TEST_SUITES["mikasa_intercept"] = {
    "tasks": [
        task
        for task in TEST_SUITES["mikasa"]["tasks"]
        if "Intercept" in task["task_name"] and "Grab" not in task["task_name"]
    ]
}

for task_name, prompt in TASK_PROMPTS.items():
    TEST_SUITES[task_name] = {
        "tasks": [
            {
                "task_name": task_name,
                "env_name": task_name,
                "baseline_prompt": False,
                "prompt": prompt,
            }
        ]
    }


def center_crop(image, batch_size=1, crop_scale=0.9, return_pil_image=False):
    image = Image.fromarray(image)
    image = image.convert("RGB")

    # Convert to TF Tensor and record original data type (should be tf.uint8)
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    # Convert to data type tf.float32 and values between [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Crop and then resize back to original size
    # image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert back to PIL Image
    image = Image.fromarray(image.numpy())
    image = image.convert("RGB")

    if return_pil_image:
        return image

    image = np.array(image)

    return image


def infer_batch(images, prompts, model):
    """Infer a batch of samples."""
    batch_size = len(images)
    assert len(prompts) == batch_size, "Number of prompts must match number of images!"

    device = torch.cuda.current_device()
    inputs = {"task": prompts}
    # images = (images / 255.0).clip(0, 1)  # Ensure image is in [0, 1] range
    # images = einops.rearrange(images, "b h w c -> b c h w")
    inputs["observation.images.image"] = images
    inputs["observation.state"] = torch.zeros((batch_size, 8), device=device)
    with torch.no_grad():
        actions = model.select_action(images)

    actions = actions.cpu().numpy()
    return actions


def get_model(cfg):
    repo_root = cfg.repo_path
    repo_id = Path(repo_root).name

    ds_meta = LeRobotDatasetMetadata(repo_id, root=repo_root)
    policy = SmolVLAPolicy.from_pretrained(
        cfg.pretrained_checkpoint, dataset_stats=ds_meta.stats
    )
    return policy


@draccus.wrap()
def entry_point(cfg: GenerateConfig) -> None:
    eval_mikasa(cfg)


def eval_mikasa(
    cfg: GenerateConfig,
    model: DINOv2wMemory | None = None,
    skip_wandb_init: bool = False,
    training_step: int = 0,
) -> None:
    enable_memory = True
    inner_lr = 0.01
    action_dim = 7
    chunk_size = 10
    # pretrained_checkpoint = "logs/2026-01-20_01-05-51_fztjrr6x/vit_memory_mikasa_step_370.safetensors"
    training_log_dir = Path(cfg.pretrained_checkpoint).parent
    config_path = training_log_dir / "config.yaml"
    training_cfg = OmegaConf.load(str(config_path))
    print("Loaded model config:")
    print(type(training_cfg))
    training_cfg = OmegaConf.to_container(training_cfg, resolve=True)

    print(model)
    model_cfg = ViTMemoryConfig(
        **training_cfg["model_config"],
    )
    print("model_cfg",model_cfg)
    model = DINOv2wMemory(
        config=model_cfg,
    )
    model.to("cuda")
    model.eval()
    model_is_newly_loaded = True

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb and not skip_wandb_init:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    num_tasks_in_suite = len(TEST_SUITES[cfg.task_suite_name]["tasks"])

    # Get expected image dimensions
    # resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    plot_data = {}

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        env_name = TEST_SUITES[cfg.task_suite_name]["tasks"][task_id]["env_name"]
        task_name = TEST_SUITES[cfg.task_suite_name]["tasks"][task_id]["task_name"]

        plot_data[task_name] = defaultdict(list)
        print(f"Running task {task_name}...")
        num_envs = cfg.num_envs
        env_kwargs_rgb = dict(
            num_envs=num_envs,
            obs_mode="rgb",
            control_mode="pd_ee_delta_pose",
            render_mode="all",
            sim_backend="gpu",
            reward_mode="normalized_dense",
        )
        unnorm_key = ""  # Action un-normalization key for OpenVLA

        env = gym.make(env_name, **env_kwargs_rgb)
        state_wrappers_list, episode_timeout = env_info(env_name)
        print(f"Episode timeout: {episode_timeout}")
        for wrapper_class, wrapper_kwargs in state_wrappers_list:
            env = wrapper_class(env, **wrapper_kwargs)

        # Start episodes
        task_episodes, task_successes = 0, 0
        dist_to_target = []
        all_rewards = []
        for episode_idx in tqdm.trange(0, cfg.num_trials_per_task, num_envs):
            # Reset environment
            obs, info = env.reset()

            colors = None
            if TEST_SUITES[cfg.task_suite_name]["tasks"][task_id]["baseline_prompt"]:
                oracle_info = [int(x) for x in info["oracle_info"].cpu()]

                if "RememberColor3" in env_name:
                    colors = [["red", "green", "blue"][x] for x in oracle_info]
                elif "RememberColor9" in env_name:
                    # 7 orange, 4 pink, 8 teal, 0 red, 6 maroon, 2 blue, 5 cyan, 3 yellow, 1 green
                    colors = [
                        [
                            "red",
                            "green",
                            "blue",
                            "yellow",
                            "purple",
                            "cyan",
                            "maroon",
                            "orange",
                            "teal",
                        ][x]
                        for x in oracle_info
                    ]
                prompts = [f"Touch the {color} cube" for color in colors]
            else:
                prompts = [
                    TEST_SUITES[cfg.task_suite_name]["tasks"][task_id]["prompt"]
                ] * num_envs

            log_file.write(f"\nTask: {prompts}\n")

            # Setup
            t = 0
            replay_images = [[] for _ in range(num_envs)]
            max_steps = 500  # default max steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            terminated_flags = np.array([False] * num_envs)
            truncated_flags = np.array([False] * num_envs)
            success_flags = np.array([False] * num_envs)
            final_rewards = np.zeros(num_envs)
            final_distances = np.zeros(num_envs)

            # reset memory at the begining of the episode
            model.reset_memory()
            model.reset_action_cache()
            while t < max_steps + cfg.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    action = env.action_space.sample()
                    action = np.zeros(action.shape)
                    obs, reward, terminated, truncated, info = env.step(action)
                    images = obs["sensor_data"]["base_camera"]["rgb"]
                    for i in range(num_envs):
                        replay_images[i].append(images[i].detach().clone())
                    
                    if t == 5:
                        actions = infer_batch(
                            images=images,
                            prompts=prompts,
                            model=model
                        )
                        model.reset_action_cache()
                    t += 1
                    continue

                # Get observation image
                images = obs["sensor_data"]["base_camera"]["rgb"]

                # Save preprocessed image for replay video
                for i in range(num_envs):
                    replay_images[i].append(images[i].detach().clone())

                if model_is_newly_loaded:
                    # run a inference to force initialisation of the memories
                    actions = infer_batch(
                        images=images,
                        prompts=prompts,
                        model=model
                    )

                    # reload the safetensors weights to fill the memory initialisation values
                    model.load(cfg.pretrained_checkpoint)

                    # reset the memory again
                    model.reset_memory()
                    model.reset_action_cache()
                    model_is_newly_loaded = False

                # query VLA model for action
                actions = infer_batch(
                    images=images,
                    prompts=prompts,
                    model=model,
                )
                actions = torch.from_numpy(actions)
                actions = actions * cfg.model_action_scale
                obs, reward, terminated, truncated, info = env.step(actions)

                for i in range(num_envs):
                    if terminated[i].cpu().numpy():
                        terminated_flags[i] = True
                        final_rewards[i] = reward[i].cpu().numpy()
                        if (
                            "reward_dict" in info
                            and "tcp_to_obj_dist" in info["reward_dict"]
                        ):
                            final_distances[i] = (
                                info["reward_dict"]["tcp_to_obj_dist"][i].cpu().numpy()
                            )
                        else:
                            final_distances[i] = -1
                    if truncated[i].cpu().numpy():
                        truncated_flags[i] = True
                        final_rewards[i] = reward[i].cpu().numpy()
                        if (
                            "reward_dict" in info
                            and "tcp_to_obj_dist" in info["reward_dict"]
                        ):
                            final_distances[i] = (
                                info["reward_dict"]["tcp_to_obj_dist"][i].cpu().numpy()
                            )
                        else:
                            final_distances[i] = -1
                    if info["success"][i].cpu().numpy():
                        success_flags[i] = True

                if all(terminated_flags | truncated_flags):
                    # print("All environments terminated or truncated, ending episode.")
                    break
                t += 1

                # except Exception as e:
                #     print(f"Caught exception: {e}")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_successes += np.sum(success_flags)
            total_successes += np.sum(success_flags)

            # Save a replay video of the episode
            for i in range(num_envs):
                task_episodes += 1
                total_episodes += 1

                mp4_path = None
                if cfg.log_rollout_videos:
                    mp4_path = save_rollout_video(
                        replay_images[i],
                        total_episodes,
                        success=terminated_flags[i],
                        task_description=task_name,
                        log_file=log_file,
                    )

                if cfg.use_wandb:
                    rollout_video_topic = (
                        f"rollout_video/{task_name}"
                        if colors is None
                        else f"rollout_video/{task_name}/{colors[i]}"
                    )

                    plot_data[task_name]["episode_idx"].append(task_episodes - 1)
                    plot_data[task_name]["success"].append(success_flags[i])
                    plot_data[task_name]["distance_to_target"].append(
                        final_distances[i]
                    )
                    plot_data[task_name]["reward"].append(final_rewards[i])

                    wandb_data = {
                        f"task/{task_name}/distance_to_target": final_distances[i],
                        f"task/{task_name}/reward": final_rewards[i],
                        f"task/{task_name}/episode_idx": task_episodes - 1,
                    }
                    if mp4_path is not None:
                        wandb_data[rollout_video_topic] = wandb.Video(
                            mp4_path, format="mp4"
                        )
                    wandb.log(wandb_data)

                dist_to_target.append(final_distances[i])
                all_rewards.append(final_rewards[i])

            # Log current results
            # print(f"Success: {terminated}")
            print(f"# episodes completed so far: {total_episodes}")
            print(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )
            log_file.write(f"Success: {terminated}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()

        # Log final results
        avg_dist_to_target = np.mean(dist_to_target)
        average_reward = np.mean(all_rewards)
        task_success_rate = float(task_successes) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)

        print(f"Current task success rate: {task_success_rate}")
        print(f"Current total success rate: {total_success_rate}")
        print(f"Average Distance to target: {avg_dist_to_target}")
        print(f"Average Reward: {average_reward}")
        log_file.write(f"Current task success rate: {task_success_rate}\n")
        log_file.write(f"Current total success rate: {total_success_rate}\n")
        log_file.write(f"Average Distance to target: {avg_dist_to_target}\n")
        log_file.write(f"Average Reward: {average_reward}\n")
        log_file.flush()

        if cfg.use_wandb:
            wandb.log(
                {
                    f"task/{task_name}/success_rate": task_success_rate,
                    f"task/{task_name}/avg_distance_to_target": avg_dist_to_target,
                    f"task/{task_name}/avg_reward": average_reward,
                }
            )

    if cfg.use_wandb and cfg.log_performance_graphs:

        def create_boxplot(metric_name: str, plot_data_key: str):
            """Return a Plotly Figure with grouped box-and-whisker plots."""
            labels = list(plot_data.keys())

            fig = go.Figure()

            # one trace per label keeps colours & hover labels tidy
            for lbl in labels:
                fig.add_trace(
                    go.Box(
                        y=plot_data[lbl][plot_data_key],
                        name=lbl,
                        boxpoints="outliers",  # show outliers, mimic Matplotlib default
                    )
                )

            fig.update_layout(
                xaxis_title=metric_name,
                boxmode="group",  # group traces side-by-side
                height=300,
                width=400,
                margin=dict(l=40, r=20, t=20, b=60),
                template="simple_white",  # clean background like most Matplotlib styles
            )

            if training_step != 0:
                fig.update_layout(title_text=f"Training Step: {training_step}")

            return fig

        def create_bar_plot(
            metric_name: str, plot_data_key: str, agg_fn=np.mean
        ):  # same default as before
            """Return a Plotly Figure with a bar chart of aggregated values."""
            labels = list(plot_data.keys())
            values = [agg_fn(plot_data[lbl][plot_data_key]) for lbl in labels]

            fig = go.Figure(data=[go.Bar(x=labels, y=values)])

            fig.update_layout(
                xaxis_title=metric_name,
                yaxis_title=agg_fn.__name__.capitalize(),
                height=300,
                width=400,
                margin=dict(l=40, r=20, t=20, b=60),
                template="simple_white",
            )
            fig.update_xaxes(tickangle=45)

            if training_step != 0:
                fig.update_layout(title_text=f"Training Step: {training_step}")

            return fig

        wandb.log(
            {
                "task_summary/reward_plot": create_boxplot(
                    "Reward", plot_data_key="reward"
                ),
                "task_summary/success_rate_plot": create_bar_plot(
                    "Success Rate", plot_data_key="success"
                ),
                "task_summary/distance_to_target_plot": create_boxplot(
                    "Distance to Target", plot_data_key="distance_to_target"
                ),
            }
        )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "total/success_rate": float(total_successes) / float(total_episodes),
                "total/num_episodes": total_episodes,
                "total/current_model_step": int(
                    os.environ.get("CURRENT_TRAINING_STEP", 0)
                ),
            }
        )
        wandb.save(local_log_filepath)

    # reset memory at the end of the run
    model.reset_memory()


if __name__ == "__main__":
    entry_point()
