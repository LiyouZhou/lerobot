import torch.nn as nn
from lerobot.policies.smolvla.memory.module import MemoryModule
import timm
from tqdm import trange


class ViTwMemory(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        pretrained=True,
        memory_size=768,
        inner_lr=0.4,
        decay_factor=0.99,
        num_classes=10,
        enable_memory=True,
    ):
        super(ViTwMemory, self).__init__()
        self.enable_memory = enable_memory
        self.encoder = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0,  # No classification head
            global_pool="",
        )
        self.encoder.reset_classifier(0)

        if self.enable_memory:
            self.memory = MemoryModule(
                hidden_size=memory_size,
                inner_learning_rate=inner_lr,
                decay_factor=decay_factor,
            )
        self.prediction_head = nn.Sequential(   
            nn.LazyLinear(memory_size),
            nn.ReLU(),
            nn.LazyLinear(num_classes),
        )
        num_layers = 1
        hidden_dim = memory_size
        num_heads = 4
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, x):
        features = self.encoder(x)  # (batch_size, embed_len, hidden_dim)
        # print("features.shape", features.shape)
        if self.enable_memory:
            out_features = self.memory(features)  # (batch_size, embed_len, hidden_dim)
        else:
            out_features = features
        # print("out_features.shape", out_features.shape)
        transformer_out = self.transformer(out_features)  # (batch_size, embed_len, hidden_dim)
        # print("transformer_out.shape", transformer_out.shape)
        flattened_out_features = transformer_out.view(transformer_out.size(0), -1)
        # print("flattened_out_features.shape", flattened_out_features.shape)
        out = self.prediction_head(flattened_out_features)
        # print("out.shape", out.shape)
        return out


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

    model = ViTwMemory(enable_memory=enable_memory)
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
    wandb.init(project="vit-memory", config=config)

    main_pbar = trange(n_steps)
    for i in main_pbar:
        os.environ["TRAINING_STEP"] = str(i)

        gt = []
        if enable_memory:
            model.memory.reset_memory()
        pbar = trange(episode_length, position=1, leave=False)
        for j in pbar:
            data = next(dataloader_iter)
            imgs, labels = data

            imgs = nn.functional.interpolate(
                imgs, size=(224, 224), mode="bilinear", align_corners=False
            )
            imgs = imgs.repeat(1, 3, 1, 1)  # Convert to 3 channels

            # print("imgs.shape", imgs.shape)

            if gt == []:
                gt = labels

            pred = model(imgs.to("cuda"))
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
                    f"train/inner_loss/{j}": model.memory.last_inner_loss if enable_memory else 0.0,
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
