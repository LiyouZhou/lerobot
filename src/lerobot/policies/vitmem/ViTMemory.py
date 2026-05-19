from lerobot.policies.vitmem.module import MemoryModule
from lerobot.policies.vitmem.slot_memory import SlotMemory
from lerobot.policies.vitmem.configuration_vitmem import ViTMemoryConfig
from lerobot.policies.pretrained import PreTrainedPolicy

import torch
import torch.nn as nn
from einops import rearrange, repeat
from tqdm import trange
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoImageProcessor, AutoModel
from safetensors.torch import load_file
from dataclasses import dataclass, asdict
import os
from torchvision.transforms import v2
from scipy.spatial.transform import Rotation as R
from sentence_transformers import SentenceTransformer


def normalize(x, min_val, max_val):
    min_val = min_val[: x.shape[-1]]
    max_val = max_val[: x.shape[-1]]
    return (x - min_val.to(x.device)) / (max_val.to(x.device) - min_val.to(x.device))


def unnormalize(x, min_val, max_val):
    min_val = min_val[: x.shape[-1]]
    max_val = max_val[: x.shape[-1]]
    return x * (max_val.to(x.device) - min_val.to(x.device)) + min_val.to(x.device)


class AttentionPool(nn.Module):
    def __init__(self, hidden_size, num_output_tokens=1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_output_tokens, hidden_size))
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

    def forward(self, x):  # x: [batch, embed_len, hidden_size]
        q = self.query.expand(
            x.size(0), -1, -1
        )  # [batch, num_output_tokens, hidden_size]
        out, _ = self.attn(q, x, x)

        return rearrange(out, "b n h -> b (n h)")


class PredictionHead(nn.Module):
    def __init__(self, config: ViTMemoryConfig):
        super(PredictionHead, self).__init__()

        self.input_projection = nn.Sequential(
            nn.LazyLinear(config.memory_size * 2), nn.ReLU()
        )

        num_fc_layers = 2
        self.fc_layers = nn.ModuleList(
            nn.LazyLinear(config.memory_size * 2) for _ in range(num_fc_layers)
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.memory_size * 2),
            nn.LazyLinear(config.chunk_size * config.action_dim),
        )

    def forward(self, x):
        if not hasattr(self, "layer_norm1"):
            self.layer_norm1 = nn.LayerNorm(x.size(-1)).to(x.device)

        x = self.layer_norm1(x)
        x = self.input_projection(x)

        # resnet-style skip connections for fc layers
        for fc in self.fc_layers:
            x = x + fc(x)

        out = self.output_projection(x)

        return out


class InputFeatureProjection(nn.Module):
    def __init__(self, config: ViTMemoryConfig):
        super(InputFeatureProjection, self).__init__()
        self.cfg = config
        num_heads = 4
        hidden_dim = self.cfg.memory_size
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.pre_attn_layer_norm = nn.LayerNorm(hidden_dim)
        expansion_factor = 4
        self.pre_ffn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

    def forward(self, features):
        normalized_features = self.pre_attn_layer_norm(features)
        attn_out, _ = self.attn(
            normalized_features, normalized_features, normalized_features
        )
        attn_out += normalized_features
        normed_attn_out = self.pre_ffn_layer_norm(attn_out)
        features = normed_attn_out + self.ffn(normed_attn_out)

        return features


class MultiLayerDecoderWithMemory(PreTrainedPolicy):
    config_class = ViTMemoryConfig
    name = "vitmem"

    def __init__(
        self,
        config: ViTMemoryConfig,
        **kwargs,
    ):
        super(MultiLayerDecoderWithMemory, self).__init__(config)
        self.cfg = config

        for i in range(self.cfg.num_layers):
            layer = DecoderWithMemory(config)
            setattr(self, f"layer_{i}", layer)

        self.input_projection = InputFeatureProjection(config)
        self.prediction_head = PredictionHead(config)

        # constant value
        dataset_metadata = None
        for key in ("state", "action"):
            for bound in ("min", "max"):
                self.register_buffer(
                    f"{key}_{bound}",
                    (
                        torch.tensor(dataset_metadata[key][bound])
                        if dataset_metadata
                        else torch.zeros(
                            self.cfg.action_dim
                            if key == "action"
                            else self.cfg.state_dim
                        )
                    ),
                )

        self.state_proj = (
            nn.Linear(
                self.cfg.state_dim,
                self.cfg.memory_size * self.cfg.num_state_tokens,
            )
            if self.cfg.proprioception
            else None
        )  # state_dim -> memory_size

        if self.cfg.pooling_method == "attention":
            self.attention_pool = AttentionPool(self.cfg.memory_size, 2)

        self.model_initialised = False

    def get_optim_params(self) -> dict:
        return self.parameters()

    def predict_action_chunk(self, x, state=None, features=None):
        out, features = self.forward(x, state, features)
        return out, features

    def reset(self):
        self.reset_memory()
        self.reset_action_cache()

    def preprocess(self, images):
        raise NotImplementedError("Subclasses should implement this method.")

    def encode(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x, state=None, features=None):
        # When called from the training loop, x is a batch dict with "action" key
        if isinstance(x, dict):
            input_dict = x
            batch_size = input_dict["observation.images.image"].shape[0]

            memory_reset_mask = torch.zeros(
                [batch_size], dtype=torch.bool
            )  # default no reset
            loss_mask = torch.ones([batch_size], dtype=torch.bool)
            if "frame_index" in input_dict.keys():
                if (
                    hasattr(self, "last_frame_indices")
                    and input_dict["frame_index"].shape == self.last_frame_indices.shape
                ):
                    memory_reset_mask = (
                        input_dict["frame_index"] < self.last_frame_indices
                    )
                    loss_mask = self.last_frame_indices != input_dict["frame_index"]
                else:
                    memory_reset_mask = input_dict["frame_index"] == 0
                self.last_frame_indices = input_dict["frame_index"]

            if torch.any(memory_reset_mask):
                self.reset_memory(
                    memory_reset_mask
                )  # reset memory at the start of each episode

            main_image = input_dict["observation.images.image"]
            if main_image.dim() == 5 and main_image.shape[1] == 1:
                main_image = main_image.squeeze(1)

            secondary_image = input_dict["observation.images.image2"]
            if secondary_image.dim() == 5 and secondary_image.shape[1] == 1:
                secondary_image = secondary_image.squeeze(1)

            x = torch.cat([main_image, secondary_image], dim=1)
            state = (
                input_dict["observation.state"]
                if "observation.state" in input_dict
                else None
            )
            if state is not None and state.dim() == 3 and state.shape[1] == 1:
                state = state.squeeze(1)

            out, pooled_features = self._forward_core(
                x,
                state,
                features,
                language_instruction=input_dict.get("task", None),
            )

            # Training mode: compute loss and return (scalar_loss, info_dict)
            if "action" in input_dict and input_dict["action"] is not None:
                gt_action = input_dict["action"]

                loss = self.compute_loss(
                    out, gt_action, loss_mask=loss_mask, normalize_gt=False
                )
            else:
                loss = torch.tensor(0.0, device=out.device)

            return loss, {"l1_loss": loss.item(), "pred": out}

        if self.cfg.proprioception and state is None:
            raise ValueError(
                "Proprioception is enabled but no state input is provided."
            )

        out, _ = self._forward_core(x, state, features, language_instruction=None)
        loss = torch.tensor(0.0, device=out[0].device)
        return loss, {"l1_loss": loss.item(), "pred": out}

    def _forward_core(self, x, state=None, features=None, language_instruction=None):
        if self.cfg.proprioception and state is None:
            raise ValueError(
                "Proprioception is enabled but no state input is provided."
            )

        batch_size = x.shape[0]

        if features is None:
            main_image = x[:, :3]
            if self.cfg.main_camera_only:
                images = main_image
            else:
                # concatinate main and secondary images along the batch dimension for joint processing
                secondary_image = x[:, 3:6]
                images = torch.cat(
                    [main_image, secondary_image], dim=0
                )  # (2*batch_size, 3, H, W)

            # Process and encode
            processed = self.preprocess(images)
            processed_cuda = processed.to("cuda")
            features = self.encode(
                processed_cuda
            )  # (2*batch_size, embed_len, hidden_dim)
        else:
            if self.cfg.main_camera_only:
                features = features[:batch_size]

        # pool vision tokens if specified in config
        features = (
            features[self.cfg.vision_token_range[0] : self.cfg.vision_token_range[1]]
            if self.cfg.vision_token_range
            else features
        )
        if self.cfg.vision_token_pooling_method == "mean":
            features = features.mean(dim=1, keepdim=True)
        elif self.cfg.vision_token_pooling_method == "max":
            features, _ = features.max(dim=1, keepdim=True)

        # split back into two tensors
        if self.cfg.main_camera_only:
            features_list = [features]
        else:
            features_list = [
                features[:batch_size],
                features[batch_size:],
            ]

        if language_instruction is not None:
            instruction_features = self.encode_task_instruction(language_instruction)
            instruction_features = rearrange(
                instruction_features, "b s -> b 1 s"
            )  # (batch_size, 1, hidden_dim)
            features_list.append(instruction_features)

        if state is not None and self.state_proj is not None:
            # state = self.normalize_state(state).to(x.device)
            state = state.to(x.device)
            state_features = self.state_proj(state)
            state_features = rearrange(
                state_features, "b (n s) -> b n s", n=self.cfg.num_state_tokens
            )
            features_list.append(state_features)

        features = torch.cat(features_list, dim=1)

        projected_features = self.input_projection(features)
        pooled_features = projected_features.mean(dim=1, keepdim=True)
        features = pooled_features

        for i in range(self.cfg.num_layers):
            layer = getattr(self, f"layer_{i}")
            features = layer(features)

        if self.cfg.pooling_method == "attention":
            pooled_features = self.attention_pool(features)
        elif self.cfg.pooling_method == "mean":
            pooled_features = features.mean(dim=1)
        elif self.cfg.pooling_method == "max":
            pooled_features, _ = features.max(dim=1)
        else:
            pooled_features = features

        # print("pooled_features.shape", pooled_features.shape)
        out = self.prediction_head(pooled_features)
        # print("out.shape", out.shape)

        if not self.model_initialised and os.path.exists(self.cfg.pre_trained_weights):
            print(f"Loading pre-trained weights from {self.cfg.pre_trained_weights}")
            self.load(self.cfg.pre_trained_weights)
            self.model_initialised = True
            self.reset_memory()
            self.reset_action_cache()
            return self._forward_core(x, state, features, language_instruction)

        return out, pooled_features

    def compute_loss(self, pred, gt, loss_mask=None, normalize_gt=True):
        if normalize_gt:
            normalized_action = self.normalize(
                gt[:, :, : self.cfg.action_dim],
            )
        else:
            normalized_action = gt[:, :, : self.cfg.action_dim]

        normalized_action = normalized_action.to("cuda")
        pred = pred.view(-1, self.cfg.chunk_size, self.cfg.action_dim)

        # If GT action is all zeros for a timestep, mask it out from the loss
        # action: (batch, chunk_size, action_dim)
        mask = (gt.abs().sum(dim=-1) != 0).to(pred.device)  # (batch, chunk)
        if loss_mask is not None:
            loss_mask = repeat(loss_mask, "b -> b n", n=mask.shape[1])
            mask = mask & loss_mask.to(pred.device)
        abs_err = torch.abs(pred - normalized_action)  # (batch, chunk, action_dim)
        masked_abs_err = abs_err * mask.unsqueeze(-1).float()

        num_unmasked = mask.sum() * pred.shape[-1]  # scalar tensor
        if num_unmasked == 0:
            return torch.tensor(0.0, device=pred.device)

        loss = masked_abs_err.sum() / num_unmasked
        return loss

    def compute_mse(self, pred, gt):
        pred = pred.view(-1, self.cfg.chunk_size, self.cfg.action_dim)

        unnormalized_pred = self.unnormalize(
            pred.clone().detach().cpu(),
        )

        mask = (gt.abs().sum(dim=-1) != 0).to(
            unnormalized_pred.device
        )  # (batch, chunk)
        masked_unnormalized_pred = unnormalized_pred * mask.unsqueeze(-1).float()

        mse = nn.MSELoss(reduction="mean")(
            masked_unnormalized_pred.to(gt.device), gt[:, :, : self.cfg.action_dim]
        )
        return mse

    def normalize_state(self, state):
        state = state.cpu()
        min_val = self.state_min.detach().clone().cpu()
        max_val = self.state_max.detach().clone().cpu()

        min_val = min_val[: state.shape[-1]]
        max_val = max_val[: state.shape[-1]]

        zero_to_one = (state - min_val.to(state.device)) / (
            max_val.to(state.device) - min_val.to(state.device)
        )
        centred_to_zero = (zero_to_one - 0.5) * 2.0  # scale to [-1, 1]
        return centred_to_zero

    def normalize(self, x):
        x = x.cpu()
        min_val = self.action_min.detach().clone().cpu()
        max_val = self.action_max.detach().clone().cpu()

        min_val = min_val[: x.shape[-1]]
        max_val = max_val[: x.shape[-1]]
        range_val = max_val.to(x.device) - min_val.to(x.device)
        range_val = torch.where(
            range_val.abs() < 1e-8, torch.ones_like(range_val), range_val
        )
        zero_to_one = (x - min_val.to(x.device)) / range_val  # scale to [0, 1]
        centred_to_zero = (zero_to_one - 0.5) * 2.0  # scale to [-1, 1]
        return centred_to_zero

    def unnormalize(self, x):
        x = x.cpu()
        min_val = self.action_min.detach().clone().cpu()
        max_val = self.action_max.detach().clone().cpu()

        min_val = min_val[: x.shape[-1]]
        max_val = max_val[: x.shape[-1]]

        x = x / 2.0 + 0.5  # scale from [-1, 1] to [0, 1]
        return x * (max_val.to(x.device) - min_val.to(x.device)) + min_val.to(x.device)

    def reset_action_cache(self):
        self.action_cache = []

    def get_action_cache(self):
        return self.action_cache

    def select_action(self, x, state=None, world_frame=False):
        if self.action_cache == []:
            loss, info_dict = self.forward(x, state)
            pred = info_dict["pred"]
            pred = pred.view(-1, self.cfg.chunk_size, self.cfg.action_dim)

            if (self.action_min != 0.0).any():
                unnormalized_pred = self.unnormalize(pred.clone().detach().cpu())
            else:
                unnormalized_pred = pred.clone().detach().cpu()

            if world_frame and state is not None:
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()

                state = repeat(state, "b c -> b n c", n=self.cfg.chunk_size)

                unnormalized_pred[:, :, :3] += state[:, :, :3]
                unnormalized_pred_euler_angles = rearrange(
                    unnormalized_pred[:, :, 3:6], "b n c -> (b n) c"
                ).numpy()
                state_euler_angles = rearrange(state[:, :, 3:6], "b n c -> (b n) c")
                pred_rot = R.from_euler("XYZ", unnormalized_pred_euler_angles)
                state_rot = R.from_euler("XYZ", state_euler_angles)
                world_frame_rot = state_rot * pred_rot
                world_frame_euler_angles = torch.from_numpy(
                    world_frame_rot.as_euler("XYZ")
                ).to(unnormalized_pred.device)
                world_frame_euler_angles = rearrange(
                    world_frame_euler_angles,
                    "(b n) c -> b n c",
                    b=unnormalized_pred.shape[0],
                    n=self.cfg.chunk_size,
                )
                unnormalized_pred[:, :, 3:6] = world_frame_euler_angles

            for i in range(self.cfg.chunk_size):
                self.action_cache.append(unnormalized_pred[:, i, :])

        return self.action_cache.pop(0)

    def load(self, checkpoint_path):
        state_dict = load_file(checkpoint_path)
        self.load_state_dict(state_dict)

    def reset_memory(self, reset_mask=None):
        if hasattr(self, "last_frame_indices"):
            del self.last_frame_indices

        for i in range(self.cfg.num_layers):
            layer = getattr(self, f"layer_{i}")
            layer.reset_memory(mask=reset_mask)

    @classmethod
    def _load_as_safetensor(
        cls, model, model_file: str, map_location: str, strict: bool
    ):
        model.cfg.pre_trained_weights = model_file
        return model


class DecoderWithMemory(nn.Module):
    def __init__(
        self,
        config: ViTMemoryConfig,
    ):
        super(DecoderWithMemory, self).__init__()
        self.cfg = config

        if self.cfg.enable_memory:
            if self.cfg.memory_type == "slot":
                self.memory = SlotMemory(
                    embed_dim=self.cfg.memory_size, num_slots=self.cfg.memory_num_slots
                )
            elif "titan" in self.cfg.memory_type:
                self.memory = MemoryModule(
                    hidden_size=self.cfg.memory_size,
                    inner_learning_rate=self.cfg.inner_lr,
                    decay_factor=self.cfg.decay_factor,
                )
            else:
                raise ValueError(f"Unsupported memory type: {self.memory_type}")

        hidden_dim = self.cfg.memory_size
        num_heads = 4
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.pre_attn_layer_norm = nn.LayerNorm(hidden_dim)
        expansion_factor = 4
        self.pre_ffn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

    def forward(self, x):
        if self.cfg.enable_memory:
            if self.cfg.memory_type == "titan":
                # Memory update uses autograd internally; ensure x has grad even during eval
                if not x.requires_grad:
                    x = x.detach().requires_grad_(True)
                self.memory.update(x)
                out_features = self.memory.retrieve(x)
            elif self.cfg.memory_type == "slot":
                out_features = self.memory.retrieve(x)
                out_features = torch.cat([x, out_features], dim=1)
                self.memory.update(x)
        else:
            out_features = x

        normalized_out_features = self.pre_attn_layer_norm(out_features)
        transformer_out, _ = self.attn(
            normalized_out_features, normalized_out_features, normalized_out_features
        )
        transformer_out += normalized_out_features
        normed_transformer_out = self.pre_ffn_layer_norm(transformer_out)
        output = normed_transformer_out + self.ffn(normed_transformer_out)

        return output

    def reset_memory(self, mask=None):
        if self.cfg.enable_memory:
            self.memory.reset_memory(mask=mask)


class VisionEncoderWithMemory(nn.Module):
    def __init__(
        self,
        config: ViTMemoryConfig,
        dataset_metadata=None,
    ):
        super(VisionEncoderWithMemory, self).__init__()
        self.cfg = config

        if self.cfg.enable_memory:
            self.memory = MemoryModule(
                hidden_size=self.cfg.memory_size,
                inner_learning_rate=self.cfg.inner_lr,
                decay_factor=self.cfg.decay_factor,
            )
        self.prediction_head = nn.Sequential(
            nn.LazyLinear(self.cfg.memory_size),
            nn.ReLU(),
            nn.LazyLinear(self.cfg.chunk_size * self.cfg.action_dim),
        )
        num_layers = 1
        hidden_dim = self.cfg.memory_size
        num_heads = 4
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, batch_first=True
            ),
            num_layers=num_layers,
        )

        # constant value
        self.register_buffer(
            "action_min",
            (
                torch.tensor(dataset_metadata["action"]["min"])
                if dataset_metadata
                else torch.zeros(self.cfg.action_dim)
            ),
        )
        self.register_buffer(
            "action_max",
            (
                torch.tensor(dataset_metadata["action"]["max"])
                if dataset_metadata
                else torch.zeros(self.cfg.action_dim)
            ),
        )

        self.action_cache = []

    def preprocess(self, images):
        raise NotImplementedError("Subclasses should implement this method.")

    def encode(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x):
        # print("Input x.shape:", x.shape)
        # print(x.device)
        processed = self.preprocess(x)

        # print("Preprocessed x.shape:", processed.shape)
        # print(processed.device)
        processed_cuda = processed.to("cuda")
        # print("After to(cuda) x.shape:", processed_cuda.shape)
        features = self.encode(processed_cuda)  # (batch_size, embed_len, hidden_dim)

        features = (
            features[self.cfg.vision_token_range[0] : self.cfg.vision_token_range[1]]
            if self.cfg.vision_token_range
            else features
        )
        if self.cfg.vision_token_pooling_method == "mean":
            features = features.mean(dim=1, keepdim=True)
        elif self.cfg.vision_token_pooling_method == "max":
            features, _ = features.max(dim=1, keepdim=True)

        # print("features.shape", features.shape)
        if self.cfg.enable_memory:
            # print("retrieve memory module")
            out_features = self.memory.retrieve(features)
            out_features = torch.concat([features, out_features], dim=1)
            # print("retrieve memory module done")
            # out_features = self.memory(features)  # (batch_size, embed_len, hidden_dim)
            # print("out_features.shape", out_features.shape)
        else:
            out_features = features
        # print("out_features.shape", out_features.shape)
        # print("Passing through transformer...")
        transformer_out = self.transformer(
            out_features
        )  # (batch_size, embed_len, hidden_dim)

        if self.cfg.enable_memory:
            self.memory.update(transformer_out)

        # print("Transformer output obtained.")
        # print("transformer_out.shape", transformer_out.shape)
        # mean pooling transformer output
        mean_out_features = transformer_out.mean(dim=1)
        # print("mean_out_features.shape", mean_out_features.shape)
        out = self.prediction_head(mean_out_features)
        # print("out.shape", out.shape)
        return out, mean_out_features

    def reset_action_cache(self):
        self.action_cache = []

    def get_action_cache(self):
        return self.action_cache

    def select_action(self, x):
        pred = None

        # update the memory with current observation
        if self.cfg.enable_memory:
            pred = self.forward(x)

        if self.action_cache == []:
            if not pred:
                pred, _ = self.forward(x)
            pred = pred.view(-1, self.cfg.chunk_size, self.cfg.action_dim)

            if (self.action_min != 0.0).any():
                unnormalized_pred = unnormalize(
                    pred.clone().detach().cpu(),
                    self.action_min.detach().clone().cpu(),
                    self.action_max.detach().clone().cpu(),
                )
            else:
                unnormalized_pred = pred.clone().detach().cpu()

            for i in range(self.cfg.chunk_size):
                self.action_cache.append(unnormalized_pred[:, i, :])

        return self.action_cache.pop(0)

    def load(self, checkpoint_path):
        state_dict = load_file(checkpoint_path)
        self.load_state_dict(state_dict)

    def reset_memory(self):
        if self.cfg.enable_memory:
            self.memory.reset_memory()


class DINOv2wMemory(MultiLayerDecoderWithMemory):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(DINOv2wMemory, self).__init__(*args, **kwargs)

        model_name = "facebook/dinov2-base"

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)

    def preprocess(self, images):
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=True)
        return inputs["pixel_values"]  # shape: (batch_size, 3, 224, 224)

    def encode(self, x):
        features = self.encoder(pixel_values=x)  # (batch_size, embed_len, hidden_dim)
        return features.last_hidden_state

    def freeze_encoder(self):
        self.encoder.eval()  # important for BN / dropout
        for p in self.encoder.parameters():
            p.requires_grad = False


class EUPEwMemory(MultiLayerDecoderWithMemory):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(EUPEwMemory, self).__init__(*args, **kwargs)

        REPO_DIR = "facebookresearch/eupe"
        WEIGHTS_URL = (
            "https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt"
        )

        # EUPE ViT models pretrained on web images
        self.encoder = torch.hub.load(REPO_DIR, "eupe_vitb16", weights=WEIGHTS_URL)

        self.sentence_transsformer = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.sentence_embedding_proj = nn.Linear(384, self.cfg.memory_size)

        def make_transform(resize_size: int = 256):
            to_tensor = v2.ToImage()
            resize = v2.Resize((resize_size, resize_size), antialias=True)
            to_float = v2.ToDtype(torch.float32, scale=True)
            normalize = v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            return v2.Compose([to_tensor, resize, to_float, normalize])

        self.processor = make_transform()

    def preprocess(self, images):
        inputs = self.processor(images)
        return inputs

    def encode(self, x):
        with torch.inference_mode():
            outputs = self.encoder.forward_features(x)

        clstoken, patchtokens = (
            outputs["x_norm_clstoken"].detach().clone().to(x.device),
            outputs["x_norm_patchtokens"].detach().clone().to(x.device),
        )

        return torch.cat([clstoken.unsqueeze(1), patchtokens], dim=1)

    def freeze_encoder(self):
        self.encoder.eval()  # important for BN / dropout
        for p in self.encoder.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.freeze_encoder()
        self.sentence_transsformer.eval()
        for p in self.sentence_transsformer.parameters():
            p.requires_grad = False

        return self

    def encode_task_instruction(self, instruction_sentences):
        with torch.inference_mode():
            embeddings = self.sentence_transsformer.encode(
                instruction_sentences,
                output_value="sentence_embedding",
                convert_to_tensor=True,
            )
        # SentenceTransformer may return inference-mode tensors; detach+clone makes a normal tensor.
        embeddings = (
            embeddings.detach().clone().to(self.sentence_embedding_proj.weight.device)
        )
        features = self.sentence_embedding_proj(embeddings)

        return features


class ViTwMemory(VisionEncoderWithMemory):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        pretrained=True,
        **kwargs,
    ):
        super(ViTwMemory, self).__init__(**kwargs)
        self.encoder = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0,  # No classification head
            global_pool="",
        )
        self.encoder.reset_classifier(0)
        self.processor = create_transform(
            **resolve_data_config(self.encoder.pretrained_cfg, model=self.encoder)
        )

    def preprocess(self, images):
        inputs = self.processor(images)
        return inputs  # shape: (batch_size, 3, 224, 224)

    def encode(self, x):
        features = self.encoder(x)  # (batch_size, embed_len, hidden_dim)
        return features


if __name__ == "__main__":
    import torch
    from torch.utils.data import Dataset
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    from torchvision.transforms import Resize
    import matplotlib.pyplot as plt
    import wandb
    import os

    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    lr = 0.00001
    batch_size = 32
    episode_length = 5
    enable_memory = True
    cfg = ViTMemoryConfig(
        enable_memory=enable_memory,
        main_camera_only=True,
    )

    model = EUPEwMemory(config=cfg)
    model.to("cuda")
    model.train()
    model.encoder.eval()

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=lr,
    )

    dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    dataloader_iter = iter(dataloader)

    n_steps = 10000

    loss_window = []
    accuracy_window = []

    config = {
        "lr": lr,
        "batch_size": batch_size,
        "episode_length": episode_length,
        "model_name": "vit_base_patch16_224",
        "enable_memory": enable_memory,
    }
    wandb.init(project="vit-memory-selftest", config=config)

    main_pbar = trange(n_steps)
    for i in main_pbar:
        os.environ["TRAINING_STEP"] = str(i)

        gt = []
        if enable_memory:
            model.reset_memory()
        pbar = trange(episode_length, position=1, leave=False)
        for j in pbar:
            data = next(dataloader_iter)
            imgs, labels = data

            # imgs = nn.functional.interpolate(
            #     imgs, size=(224, 224), mode="bilinear", align_corners=False
            # )
            imgs = imgs.repeat(1, 3, 1, 1)  # Convert to 3 channels

            # print("imgs.shape", imgs.shape)
            imgs = imgs * 255.0  # Scale to [0, 255]

            if gt == []:
                gt = labels

            pred, _ = model(imgs.to("cuda"))
            loss = nn.CrossEntropyLoss()(pred, gt.to("cuda"))

            loss.backward()

            # print("pred.shape", pred.shape)
            # print("pred", pred.argmax(dim=1))
            # print("labels", labels)
            # print("gt", gt)

            accuracy = (pred.argmax(dim=1) == gt.to("cuda")).float().mean()
            loss_value = loss.item()

            wandb.log(
                {
                    f"train/loss/{j}": loss_value,
                    f"train/accuracy/{j}": accuracy,
                },
                step=i,
            )

            loss_window.append(loss_value)
            accuracy_window.append(accuracy.item())
            window_size = 100

            average_loss = sum(loss_window[-window_size:]) / len(
                loss_window[-window_size:]
            )
            average_accuracy = sum(accuracy_window[-window_size:]) / len(
                accuracy_window[-window_size:]
            )
            loss_window = loss_window[-window_size:]
            accuracy_window = accuracy_window[-window_size:]

            main_pbar.set_postfix(
                {f"Loss:": f"{average_loss:.4f}", "Acc": f"{average_accuracy:.4f}"}
            )

        # only update weights after each episode
        optimizer.step()
        optimizer.zero_grad()
