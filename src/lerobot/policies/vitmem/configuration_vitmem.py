from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("vitmem")
@dataclass
class ViTMemoryConfig(PreTrainedConfig):
    memory_size: int = 768
    inner_lr: float = 0.4
    decay_factor: float = 0.99
    enable_memory: bool = True
    n_action_steps: int = 5
    action_dim: int = 7
    state_dim: int = 25
    chunk_size: int = 10
    vision_token_range: tuple[int, int] = (
        0,
        10000000,
    )  # [start, end) indices for selecting tokens from the vision encoder output
    vision_token_pooling_method: str | None = (
        None  # "mean", "max", or None (no pooling, use all tokens)
    )
    num_layers: int = 1
    memory_type: str = "titan"
    memory_num_slots: int = 10  # Only used if memory_type is "slot"
    pre_trained_weights: str = "/does/not/exist/weights.safetensors"
    num_state_tokens: int = 16
    proprioception: bool = False
    pooling_method: str = "attention"  # "attention", "mean", "max", or None
    attn_pool_num_tokens: int = 8
    main_camera_only: bool = (
        False  # If True, only use the main camera image and ignore the secondary camera
    )

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.01
    vision_backbone: str = "eupe"
    train_vision_encoder: bool = False
    vision_encoder_lr_multiplier: float = 1.0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    vision_token_cls_only: bool = (
        False  # If True, only use the [CLS] token from the vision encoder output as the visual feature
    )
    pre_transformer_pooling_method: str | None = (
        "mean"  # "mean", "max", or None (no pooling, use all tokens)
    )
    pre_transformer_project: bool = (
        True  # If True, project the features before the transformer
    )

    def __post_init__(self):
        super().__post_init__()

    def validate_features(self) -> None:
        pass

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
