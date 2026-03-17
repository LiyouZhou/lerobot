import torch
import torch.nn as nn
from lerobot.policies.smolvla.memory.module import MemoryModule
import timm
from tqdm import trange
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoImageProcessor, AutoModel
from safetensors.torch import load_file
from dataclasses import dataclass, asdict
import os


def normalize(x, min_val, max_val):
    min_val = min_val[: x.shape[-1]]
    max_val = max_val[: x.shape[-1]]
    return (x - min_val.to(x.device)) / (max_val.to(x.device) - min_val.to(x.device))


def unnormalize(x, min_val, max_val):
    min_val = min_val[: x.shape[-1]]
    max_val = max_val[: x.shape[-1]]
    return x * (max_val.to(x.device) - min_val.to(x.device)) + min_val.to(x.device)


@dataclass
class ViTMemoryConfig:
    memory_size: int = 768
    inner_lr: float = 0.4
    decay_factor: float = 0.99
    enable_memory: bool = True
    n_action_steps: int = 5
    action_dim: int = 7
    chunk_size: int = 10
    vision_token_range: tuple[int, int] = (
        0,
        10000000,
    )  # [start, end) indices for selecting tokens from the vision encoder output
    vision_token_pooling_method: str | None = (
        None  # "mean", "max", or None (no pooling, use all tokens)
    )
    num_layers: int = 1

class MultiLayerDecoderWithMemory(nn.Module):
    def __init__(
        self,
        config: ViTMemoryConfig,
        dataset_metadata=None,
    ):
        super(MultiLayerDecoderWithMemory, self).__init__()
        self.cfg = config

        for i in range(self.cfg.num_layers):
            layer = DecoderWithMemory(config)
            setattr(self, f"layer_{i}", layer)

        self.prediction_head = nn.Sequential(
            nn.LazyLinear(self.cfg.memory_size),
            nn.ReLU(),
            nn.LazyLinear(self.cfg.chunk_size * self.cfg.action_dim),
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

    def preprocess(self, images):
        raise NotImplementedError("Subclasses should implement this method.")

    def encode(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x):
        processed = self.preprocess(x)
        processed_cuda = processed.to("cuda")
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

        for i in range(self.cfg.num_layers):
            layer = getattr(self, f"layer_{i}")
            features = layer(features)

        mean_out_features = features.mean(dim=1)
        # print("mean_out_features.shape", mean_out_features.shape)
        out = self.prediction_head(mean_out_features)
        # print("out.shape", out.shape)
        return out, mean_out_features

    def reset_action_cache(self):
        self.action_cache = []

    def get_action_cache(self):
        return self.action_cache

    def select_action(self, x):
        if self.action_cache == []:
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
        for i in range(self.cfg.num_layers):
            layer = getattr(self, f"layer_{i}")
            layer.reset_memory()

class DecoderWithMemory(nn.Module):
    def __init__(
        self,
        config: ViTMemoryConfig,
    ):
        super(DecoderWithMemory, self).__init__()
        self.cfg = config

        if self.cfg.enable_memory:
            self.memory = MemoryModule(
                hidden_size=self.cfg.memory_size,
                inner_learning_rate=self.cfg.inner_lr,
                decay_factor=self.cfg.decay_factor,
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

    def forward(self, x):
        if self.cfg.enable_memory:
            out_features = self.memory.retrieve(x)
            out_features = torch.concat([x, out_features], dim=1)
        else:
            out_features = x

        transformer_out = self.transformer(out_features)

        if self.cfg.enable_memory:
            self.memory.update(transformer_out)

        return transformer_out
    
    def reset_memory(self):
        if self.cfg.enable_memory:
            self.memory.reset_memory()


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
        if self.action_cache == []:
            pred = self.forward(x)
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
    )

    model = DINOv2wMemory(config=cfg)
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
            optimizer.step()
            optimizer.zero_grad()

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
