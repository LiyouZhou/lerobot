import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import repeat, rearrange

from transformers.models.llama.modeling_llama import LlamaMLP, LlamaDecoderLayer

import torch.nn.functional as F


class Memory(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)


class MemoryModule(nn.Module):
    def __init__(self, hidden_size, local_update_lr: float = 1e-4):
        super().__init__()
        D = hidden_size
        self.M = nn.Parameter(torch.empty(D, D))
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
        for p in (self.M, self.w_k, self.w_v, self.w_q):
            nn.init.xavier_uniform_(p)
        self.initialised = True

    def reset_memory(self):
        self.current_M = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, L, _ = x.shape  # step: [batch_size, embed_length, hidden_size]

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
                    self.current_M = repeat(self.M, "d1 d2 -> b d1 d2", b=B).requires_grad_(
                        True
                    )  # [B, D, D]
                else:
                    self.current_M = (
                        self.current_M.detach().clone().requires_grad_(True)
                    )  # [B, D, D]

                # inner‐loop loss & gradient wrt current_M (first-order)
                K = rearrange(K, "(b l) d -> b l d", l=L)  # [B, L, D]
                pred = torch.bmm(K, self.current_M)  # [B, L, D]
                pred = rearrange(pred, "b l d -> (b l) d")  # [B * L, D]
                inner_l = F.mse_loss(pred, V)
                (gM,) = torch.autograd.grad(inner_l, self.current_M, create_graph=False)

                # one gradient step on current_M (detach to prevent second-order gradients)
                self.current_M = self.current_M - self.local_update_lr * gM.detach()

                # retrieval with the adapted memory
                Q = x_flat @ self.w_q.t()  # [B * L, D]
                Q = rearrange(Q, "(b l) d -> b l d", l=L)  # [B, L, D]
                out_half = torch.bmm(Q, self.current_M)  # [B, L, D]

        # 2) cast back to original input dtype
        return out_half.to(x.dtype)


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
    memory = MemoryModule(hidden_size=1024)
    optimizer = torch.optim.Adam(memory.parameters(), lr=1e-4)

    for iteration in range(1000):
        x = torch.randn(
            16, 20, 300, 1024
        )  # [episodes in batch, steps in episode, features]
        outputs = []

        # Reset memory at the start of each batch
        memory.current_M = None

        for step in range(x.shape[1]):  # iterate through the steps in the episodes
            out = memory(x[:, step, :, :])
            outputs.append(out)

        # Stack outputs along the step dimension
        outputs = torch.stack(outputs, dim=1)
        loss = F.mse_loss(outputs, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
