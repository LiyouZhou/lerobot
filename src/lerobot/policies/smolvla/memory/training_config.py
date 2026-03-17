from dataclasses import dataclass, field
from lerobot.policies.smolvla.memory.ViTMemory import ViTMemoryConfig


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

    # eval
    eval_steps: int = 1000
    task_suite_name: str = "mikasa_remember_color"
    model_action_scale: float = 10.0
    num_envs: int = 10
    num_trials_per_task: int = 100
