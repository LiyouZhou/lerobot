import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import repeat, rearrange

from transformers.models.llama.modeling_llama import LlamaMLP, LlamaDecoderLayer

import torch.nn.functional as F


class Memory(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)


class MLPMemory(nn.Module):
    def __init__(self, n=2, device="cuda"):
        super().__init__()

        self.B = 0
        self.L = 0
        self.D = 0
        self.n = n  # number of layers
        self.device = device

    def create_weights(self):
        self.layer_0_weights = torch.empty(
            self.B, self.D, self.D, requires_grad=True, device=self.device
        )
        self.layer_0_bias = torch.empty(
            self.B, self.L, self.D, requires_grad=True, device=self.device
        )
        self.layer_1_weights = torch.empty(
            self.B, self.D, self.D, requires_grad=True, device=self.device
        )
        self.layer_1_bias = torch.empty(
            self.B, self.L, self.D, requires_grad=True, device=self.device
        )
        self.layers = [
            self.layer_0_weights,
            self.layer_0_bias,
            self.layer_1_weights,
            self.layer_1_bias,
        ]

    def forward(self, x):
        # x is of the shape [B, L, D]
        B, L, D = x.shape
        if self.B != B:
            self.B, self.L, self.D = B, L, D
            self.create_weights()
            if not hasattr(self, "_saved_weights"):
                self.initialize_weights()
            else:
                self.reset_memory()

        # (B x L x D bmm B X D X D) + B x L x D -> B x L x D
        x = torch.bmm(x, self.layer_0_weights) + self.layer_0_bias
        x = F.gelu(x)
        x = torch.bmm(x, self.layer_1_weights) + self.layer_1_bias
        x = F.gelu(x)

        return x

    def initialize_weights(self, init_weights=None):
        if init_weights is not None:
            self._saved_weights = init_weights
            self.reset_memory()
        else:
            self._saved_weights = []
            for p in self.layers:
                if p.dim() > 1:
                    temp_state = torch.empty_like(p[0], device=self.device)
                    nn.init.xavier_uniform_(temp_state)
                    p.copy_(
                        repeat(temp_state, "d1 d2 -> b d1 d2", b=p.shape[0])
                        .clone()
                        .detach()
                    )
                    self._saved_weights.append(temp_state.clone().detach())
                else:
                    nn.init.normal_(p, mean=0.0, std=0.02)

    def get_init_weights(self):
        return self._saved_weights

    def reset_memory(self):
        for p, state in zip(self.layers, self._saved_weights):
            p.copy_(repeat(state, "d1 d2 -> b d1 d2", b=p.shape[0]).clone().detach())

    def update(self, loss, learning_rate=1e-4):
        for i, layer in enumerate(self.layers):
            assert layer.requires_grad
            grad = torch.autograd.grad(
                loss, layer, retain_graph=True, allow_unused=True
            )[0]

            if grad is not None:
                layer.data = layer.data - learning_rate * grad.detach()

                # print(f"Layer {i} | Param {name} | ", end="")
                # print(grad.abs().mean().item(), end=" ")
                # print()


class MemoryModule(nn.Module):
    def __init__(self, hidden_size, local_update_lr: float = 1e-4):
        super().__init__()
        D = hidden_size
        self.M = None
        self.w_k = nn.Parameter(torch.empty(D, D))
        self.w_v = nn.Parameter(torch.empty(D, D))
        self.w_q = nn.Parameter(torch.empty(D, D))

        self.local_update_lr = nn.Parameter(
            torch.tensor(local_update_lr)
        )  # wrap in nn.Parameter to make it trainable
        self.memory_gate = nn.Parameter(torch.ones(hidden_size) / 2)

        self.initialised = False
        self.current_M = None  # Track current memory state

    def initialize_weights(self):
        for p in (self.w_k, self.w_v, self.w_q):
            nn.init.xavier_uniform_(p)

        self.initialised = True

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

        # 1) run all inner‐loop math in half precision
        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            with torch.enable_grad():
                x_flat = rearrange(x, "b l d -> (b l) d")  # [B * L, D]

                K = x_flat @ self.w_k.t()  # [B * L, D]
                V = x_flat @ self.w_v.t()  # [B * L, D]

                # Initialize or use current memory state
                if self.current_M is None:
                    # Don't detach here - we want gradients to flow back to self.M
                    # for each sample in the batch, create a [D, D] memory matrix initialised by self.M
                    self.current_M = MLPMemory()

                # inner‐loop loss & gradient wrt current_M (first-order)
                K = rearrange(K, "(b l) d -> b l d", b=B, l=L)  # [B, L, D]
                pred = self.current_M(K)  # [B, L, D]
                pred = rearrange(pred, "b l d -> (b l) d")  # [B * L, D]
                inner_l = F.mse_loss(pred, V)
                self.current_M.update(inner_l)

                # retrieval with the adapted memory
                Q = x_flat @ self.w_q.t()  # [B * L, D]
                Q = rearrange(Q, "(b l) d -> b l d", b=B, l=L)  # [B, L, D]
                out_half = self.current_M(Q)  # [B, L, D]

        # 2) cast back to original input dtype
        return out_half.to(x.dtype)

    def batch_forward_memory(self, x):
        B, L, D = x.shape

        pred = []
        for i in range(B):
            self.current_M[i] = self.current_M[i].to(x.device)
            pred.append(self.current_M[i](x[i]))
            # print(f"x {i}: ", x[i])
            # print(f"Memory {i} output mean: ", pred[-1].abs().mean().item())
        # pred is a list of length B, each element is [L, D]

        pred = torch.stack(pred, dim=0)  # [B, L, D]

        return pred


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

    memory = MemoryModule(hidden_size=1024)
    memory.initialize_weights()
    optimizer = torch.optim.Adam(memory.parameters(), lr=1e-4)
    memory.to(device="cuda")

    for iteration in trange(1000):
        x = torch.randn(
            16, 20, 300, 1024
        )  # [episodes in batch, steps in episode, features]
        outputs = []

        # Reset memory at the start of each batch
        memory.current_M = None
        x = x.to(device="cuda")
        for step in range(x.shape[1]):  # iterate through the steps in the episodes
            out = memory(x[:, step, :, :])
            # print("x[:, step, :, :]", x[:, step, :, :])
            outputs.append(out)

        # Stack outputs along the step dimension
        outputs = torch.stack(outputs, dim=1)
        loss = F.mse_loss(outputs, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
