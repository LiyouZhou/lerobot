import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import repeat, rearrange

from transformers.models.llama.modeling_llama import LlamaMLP, LlamaDecoderLayer

import torch.nn.functional as F
import wandb
import os


class Memory(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)


class MLPMemory(nn.Module):
    def __init__(self, n=2, device="cuda"):
        super().__init__()

        self.n = n  # number of layers
        self.device = device

    def create_weights(self, batch_size, embed_len, hidden_size):
        self.B = batch_size
        self.L = embed_len
        self.D = hidden_size

        self._saved_weights = []
        self.fc0 = torch.empty(
            self.B, self.D, self.D, requires_grad=True, device=self.device
        )
        temp_state = torch.empty_like(
            self.fc0[0], device=self.device, requires_grad=True
        )
        nn.init.xavier_uniform_(temp_state)
        self.fc0 = repeat(
            temp_state.clone().detach(), "... -> b ...", b=self.fc0.shape[0]
        )
        self._saved_weights.append(temp_state.clone().detach())

        self.fc1 = torch.empty(
            self.B, self.D, self.D, requires_grad=True, device=self.device
        )
        temp_state = torch.empty_like(
            self.fc1[0], device=self.device, requires_grad=True
        )
        nn.init.xavier_uniform_(temp_state)
        self.fc1 = repeat(
            temp_state.clone().detach(), "... -> b ...", b=self.fc1.shape[0]
        )
        self._saved_weights.append(temp_state.clone().detach())

        self.fc0.requires_grad = True
        self.fc1.requires_grad = True

        for i, t in enumerate(self._saved_weights):
            self.register_buffer(f"_saved_weights_{i}", t)

    def forward(self, x):
        self.device = x.device
        B, L, D = x.shape
        if not hasattr(self, "fc0"):
            self.create_weights(batch_size=B, embed_len=L, hidden_size=D)

        if self.fc0.shape[0] != B:
            self.B = B
            self.reset_memory()

        # B x L x D bmm B X D X D -> B x L x D
        a0 = torch.bmm(x, self.fc0)
        o0 = F.gelu(a0)
        a1 = torch.bmm(o0, self.fc1)

        return a1

    def detach(self):
        if not hasattr(self, "fc0"):
            return
        self.fc0 = self.fc0.detach()
        self.fc1 = self.fc1.detach()
        self.fc0.requires_grad = True
        self.fc1.requires_grad = True

    def reset_memory(self):
        if not hasattr(self, "_saved_weights"):
            # let the first inference call create the weights
            print("memory not initialized yet, skip reset_memory")
            return

        # put the weights back to the initial state
        self.fc0 = repeat(
            self._saved_weights[0].clone().detach(), "... -> b ...", b=self.B
        )
        self.fc1 = repeat(
            self._saved_weights[1].clone().detach(), "... -> b ...", b=self.B
        )
        self.fc0.requires_grad = True
        self.fc1.requires_grad = True

        self.reset_past_surprise()

        # print("weights after reset:")
        # print(self.fc0[0].norm(p=2), self.fc1[0].norm(p=2))

        # print("past surprise after reset:")
        # if hasattr(self, "past_surprise_fc0"):
        #     print(self.past_surprise_fc0, self.past_surprise_fc1)
        # else:
        #     print("no past surprise")

    def update(self, loss, decay_factor, adaptive_lr):
        # Check if past_surprise_fc0 and past_surprise_fc1 exist, if not initialize to zeros
        if not hasattr(self, "past_surprise_fc0"):
            self.past_surprise_fc0 = torch.zeros_like(self.fc0)
            self.past_surprise_fc1 = torch.zeros_like(self.fc1)

        fc0_grad = torch.autograd.grad(
            loss, self.fc0, retain_graph=True, create_graph=True
        )[0]
        fc1_grad = torch.autograd.grad(
            loss, self.fc1, retain_graph=True, create_graph=True
        )[0]

        # remove effect of batch size on the gradient
        fc0_grad = fc0_grad * self.B
        fc1_grad = fc1_grad * self.B

        # clip gradients
        fc0_grad = torch.clamp(fc0_grad, -1.0, 1.0)
        fc1_grad = torch.clamp(fc1_grad, -1.0, 1.0)

        # self.cached_fc0_grad = fc0_grad.clone().detach()
        # self.cached_fc1_grad = fc1_grad.clone().detach()

        # print(
        #     "adaptive_lr:",
        #     adaptive_lr.mean().item(),
        #     "decay_factor:",
        #     decay_factor.mean().item(),
        # )
        # print("fc0_grad shape", fc0_grad.shape, fc1_grad.shape)
        # print("adaptive_lr shape", adaptive_lr.shape, decay_factor.shape)
        adaptive_lr = rearrange(adaptive_lr, "b () -> b 1 1", b=self.B)
        decay_factor = rearrange(decay_factor, "b () -> b 1 1", b=self.B)

        surprise_fc0 = decay_factor * self.past_surprise_fc0 - adaptive_lr * fc0_grad
        surprise_fc1 = decay_factor * self.past_surprise_fc1 - adaptive_lr * fc1_grad

        if wandb.run is not None:
            wandb.log(
                {
                    "mem_debug/fc0_grad_norm": fc0_grad.norm(p=2).mean().item(),
                    "mem_debug/fc1_grad_norm": fc1_grad.norm(p=2).mean().item(),
                    "mem_debug/decay_factor_mean": decay_factor.mean().item(),
                    "mem_debug/adaptive_lr_mean": adaptive_lr.mean().item(),
                },
                step=int(os.environ.get("TRAINING_STEP", 0)),
            )

        self.fc0 = self.fc0 + surprise_fc0
        self.fc1 = self.fc1 + surprise_fc1

        self.past_surprise_fc0 = surprise_fc0.clone().detach()
        self.past_surprise_fc1 = surprise_fc1.clone().detach()

    def reset_past_surprise(self):
        if hasattr(self, "past_surprise_fc0"):
            del self.past_surprise_fc0
        if hasattr(self, "past_surprise_fc1"):
            del self.past_surprise_fc1


class MemoryModule(nn.Module):
    def __init__(self, hidden_size, inner_learning_rate=None, decay_factor=None):
        super().__init__()
        D = hidden_size
        self.M = None
        self.w_k = nn.Parameter(torch.empty(D, D))
        self.w_v = nn.Parameter(torch.empty(D, D))
        self.w_q = nn.Parameter(torch.empty(D, D))
        self.current_M = MLPMemory()
        self.initialised = False

        class _ScaleModule(nn.Module):
            def __init__(self, factor):
                super().__init__()
                self.register_buffer("factor", torch.tensor(factor))

            def forward(self, x):
                batch_size = x.shape[0]
                ones = torch.ones((batch_size, 1), device=x.device)
                return ones * self.factor

        class _LinearAdaptor(nn.Module):
            def __init__(self):
                super().__init__()
                self.adaptor = nn.Sequential(
                    nn.LazyLinear(1),  # out_features = 1, in_features inferred lazily
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.adaptor(x)

        self.lr_adaptor = (
            _LinearAdaptor()
            if inner_learning_rate is None or inner_learning_rate == 0
            else _ScaleModule(inner_learning_rate)
        )
        self.decay_factor_generator = (
            _LinearAdaptor()
            if decay_factor is None or decay_factor == 0
            else _ScaleModule(decay_factor)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for p in (self.w_k, self.w_v, self.w_q):
            nn.init.xavier_uniform_(p)

    def reset_memory(self):
        if hasattr(self.current_M, "reset_memory") and callable(
            getattr(self.current_M, "reset_memory")
        ):
            self.current_M.reset_memory()
        elif self.current_M is None:
            pass
        else:
            raise ValueError(f"Cannot deal with memory type {type(self.current_M)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, L, D = x.shape  # step: [batch_size, embed_length, hidden_size]
        input_dtype = x.dtype
        x = x.to(dtype=self.w_k.dtype)

        # 1) run all inner‐loop math in half precision
        with torch.enable_grad():
            K = x @ self.w_k.t()  # [B, L, D]
            V = x @ self.w_v.t()  # [B, L, D]

            # Detach adapted memory from previous step
            # For the purpose of outter loop, the memory is a constant
            self.current_M.detach()

            # inner‐loop loss & gradient wrt current_M (first-order)
            pred = self.current_M(K)  # [B, L, D]
            inner_l = F.mse_loss(pred, V)
            x_flat = rearrange(x, "b l d -> b (l d)")
            self.current_M.update(
                inner_l,
                decay_factor=self.decay_factor_generator(x_flat),
                adaptive_lr=self.lr_adaptor(x_flat),
            )
            self.last_inner_loss = inner_l.item()

            # retrieval with the adapted memory
            Q = x @ self.w_q.t()  # [B, L, D]
            out_value = self.current_M(Q)  # [B, L, D]

        # self.cached_adaptive_lr = adaptive_rl.clone().detach()
        # self.cached_out_value = out_value.clone().detach()

        # All parameters are initialized after the first forward pass
        if not self.initialised:
            self.initialised = True

        return out_value.to(dtype=input_dtype)


class MemoryLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx, memory_enabled: bool = True):
        super().__init__(config, layer_idx)
        self.memory_enabled = memory_enabled
        if self.memory_enabled:
            self.neural_memory = MemoryModule(config)
            # if you were relying on HF's gradient_checkpointing flag, keep it off:
            self.gradient_checkpointing = False
            self.memory_gate = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        # --- self-attention as before ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # --- feed-forward + memory ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.memory_gate * self.neural_memory(hidden_states) + (
            1 - self.memory_gate
        ) * self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # --- pack outputs ---
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present,)

        return outputs


if __name__ == "__main__":
    from tqdm import trange
    import random
    import os

    episode_len = 10
    hidden_size = 1024
    batch_size = 16
    lr = 1e-4
    inner_lr = 1e-2
    decay_factor = 0.9
    first_order_approx = False

    final_linear_layer = nn.Linear(hidden_size * hidden_size, 4).to(device="cuda")
    memory = MemoryModule(
        hidden_size=hidden_size, inner_learning_rate=inner_lr, decay_factor=decay_factor
    ).to(device="cuda")

    config = {
        "episode_len": episode_len,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "lr": lr,
        "inner_lr": inner_lr,
        "decay_factor": decay_factor,
        "first_order_approx": first_order_approx,
    }
    wandb.init(project="memory-module", config=config)

    optimizer = torch.optim.Adam(
        list(memory.parameters()) + list(final_linear_layer.parameters()),
        lr=lr,
    )
    memory.to(device="cuda")

    loss_window = []
    accuracy_window = []
    fc0_grad_window = []
    fc1_grad_window = []
    pbar = trange(10000)
    test = False
    for iteration in pbar:
        os.environ["TRAINING_STEP"] = str(iteration)

        if test:
            memory.eval()
            batch_size = 32

        # batch_size = random.randint(14, 16)
        x = torch.randn(
            batch_size, episode_len, hidden_size, hidden_size, device="cuda"
        )  # [episodes in batch, steps in episode, features]

        # hide some privileged information in the second frame
        gt = torch.randint(0, 4, (batch_size,), device="cuda")
        for j in range(0, 1):
            for i in range(batch_size):
                x[i, j, :, :] += (
                    torch.ones(hidden_size, hidden_size, device="cuda") * gt[i]
                )

        # Reset memory at the start of each batch
        memory.reset_memory()
        x = x.to(device="cuda")
        inner_losses = []
        # iterate through the steps in the episodes

        loss = torch.tensor(0.0).to(device="cuda")
        for step in range(x.shape[1]):
            out = memory(x[:, step, :, :])
            inner_losses.append(memory.last_inner_loss)
            y_pred = rearrange(out, "b d1 d2 -> b (d1 d2)")
            # Project y_pred into logits
            logits = final_linear_layer(y_pred)

            if first_order_approx:
                loss = torch.tensor(0.0).to(device="cuda")
            # Cross entropy loss between logits and gt
            loss += F.cross_entropy(logits, torch.tensor(gt, device=x.device))
            # if not test:
            #     if step > 0:
            #         loss += F.cross_entropy(logits, torch.tensor(gt, device=x.device))
            # else:
            #     loss = torch.tensor(0.0)

            accuracy = (
                (logits.argmax(dim=-1) == torch.tensor(gt, device=x.device))
                .float()
                .mean()
                .item()
            )

            wandb.log(
                {
                    f"train/inner_loss/{step}": memory.last_inner_loss,
                    f"train/loss/{step}": loss.item(),
                    f"train/accuracy/{step}": accuracy,
                },
                step=iteration,
            )

            if step > 0:
                loss_window.append(loss.item())
                accuracy_window.append(accuracy)
                # fc0_grad_window.append(
                #     torch.mean(memory.current_M.cached_fc0_grad).item() * 1e5
                # )
                # fc1_grad_window.append(
                #     torch.mean(memory.current_M.cached_fc1_grad).item() * 1e5
                # )
                if len(loss_window) > 500:
                    loss_window.pop(0)
                    accuracy_window.pop(0)
                    # fc0_grad_window.pop(0)
                    # fc1_grad_window.pop(0)

                pbar.set_postfix(
                    {
                        "Loss": f"{sum(loss_window)/len(loss_window):.03f}",
                        "accuracy": f"{sum(accuracy_window)/len(loss_window):.03f}",
                        # "params mean": f"{torch.mean(memory.w_k).item():.03f} {torch.mean(memory.w_v).item():.03f} {torch.mean(memory.w_q).item():.03f}",
                        # "params std": f"{torch.std(memory.w_k).item():.03f} {torch.std(memory.w_v).item():.03f} {torch.std(memory.w_q).item():.03f}",
                        # "lr": f"{torch.mean(memory.cached_adaptive_lr).item():.03f}, std: {torch.std(memory.cached_adaptive_lr).item():.03f}",
                        # "fc0_grad": f"{sum(fc0_grad_window)/len(fc0_grad_window):.03f}",
                        # "fc1_grad": f"{sum(fc1_grad_window)/len(fc1_grad_window):.03f}",
                    }
                )

                average_accuracy = (
                    sum(accuracy_window) / len(accuracy_window)
                    if accuracy_window
                    else 0
                )
                # if len(accuracy_window) > 300 and average_accuracy > 0.80:
                #     test = True

            if first_order_approx:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # if not test:
        if not first_order_approx:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if iteration == 0:
            for name, param in memory.named_parameters():
                print(name, param.shape, param.requires_grad)
