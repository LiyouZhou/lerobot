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
    def __init__(self, batch_size, embed_len, hidden_size, n=2, device="cuda"):
        super().__init__()

        self.B = batch_size
        self.L = embed_len
        self.D = hidden_size
        self.n = n  # number of layers
        self.device = device

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

    def forward(self, x):
        # B x L x D bmm B X D X D -> B x L x D
        a0 = torch.bmm(x, self.fc0)
        o0 = F.gelu(a0)
        a1 = torch.bmm(o0, self.fc1)

        return a1

    def reset_memory(self):
        self.fc0 = repeat(
            self._saved_weights[0].clone().detach(), "... -> b ...", b=self.B
        )
        self.fc1 = repeat(
            self._saved_weights[1].clone().detach(), "... -> b ...", b=self.B
        )
        self.fc0.requires_grad = True
        self.fc1.requires_grad = True

    def update(self, loss, learning_rate=1e-4):
        fc0_grad = torch.autograd.grad(
            loss, self.fc0, retain_graph=True, create_graph=True
        )[0]
        fc1_grad = torch.autograd.grad(
            loss, self.fc1, retain_graph=True, create_graph=True
        )[0]

        self.fc0 = self.fc0 - learning_rate * fc0_grad
        self.fc1 = self.fc1 - learning_rate * fc1_grad


class MemoryModule(nn.Module):
    def __init__(
        self, batch_size, embed_len, hidden_size, local_update_lr: float = 1e-4
    ):
        super().__init__()
        D = hidden_size
        self.M = None
        self.w_k = nn.Parameter(torch.empty(D, D))
        self.w_v = nn.Parameter(torch.empty(D, D))
        self.w_q = nn.Parameter(torch.empty(D, D))

        # Don't detach here - we want gradients to flow back to self.M
        # for each sample in the batch, create a [D, D] memory matrix initialised by self.M
        self.current_M = MLPMemory(
            batch_size=batch_size, embed_len=embed_len, hidden_size=hidden_size
        )

        self.local_update_lr = nn.Parameter(
            torch.tensor(local_update_lr)
        )  # wrap in nn.Parameter to make it trainable
        self.memory_gate = nn.Parameter(torch.ones(hidden_size) / 2)

        self.initialised = False

        self.initialize_weights()

        self.projection_layer = nn.Parameter(torch.randn(hidden_size, hidden_size))

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

                out_half = rearrange(out_half, "b l d -> (b l) d")
                out_half = out_half @ self.projection_layer
                out_half = rearrange(out_half, "(b l) d -> b l d", b=B, l=L)

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
    from tqdm import trange

    hidden_size = 100
    batch_size = 16
    embed_len = hidden_size
    memory = MemoryModule(
        hidden_size=hidden_size, batch_size=batch_size, embed_len=embed_len
    )
    optimizer = torch.optim.Adam(memory.parameters(), lr=1e-4)
    memory.to(device="cuda")

    for param in memory.parameters():
        print(param.shape)

    for name, param in memory.named_parameters():
        print(name, param.requires_grad)

    pbar = trange(10000)
    for iteration in pbar:
        x = torch.randn(
            16, 3, hidden_size, hidden_size
        )  # [episodes in batch, steps in episode, features]
        outputs = []

        # Reset memory at the start of each batch
        memory.reset_memory()
        x = x.to(device="cuda")
        for step in range(x.shape[1]):  # iterate through the steps in the episodes
            out = memory(x[:, step, :, :])
            outputs.append(out)

        output = outputs[-1]
        # ones = torch.ones_like(output)
        y_true = x[:, 1, :, :]
        y_pred = x[:, 2, :, :]
        y_pred = F.gelu(y_pred)
        loss = 0.5 * ((y_pred - y_true) ** 2).mean()
        # print(loss.item())

        # print(torch.mean(output).item())

        # print(torch.mean(memory.memory_gate).item())

        # for v in [memory.w_k, memory.w_v, memory.w_q]:
        #     print(torch.mean(v).item(), torch.std(v).item())
        # los                                                                                                                                                                                                                            s = F.mse_loss(output, ones)

        out_mean = torch.mean(output)
        out_std = torch.std(output)

        data_mean = torch.mean(x)
        data_std = torch.std(x)

        mse_loss = F.mse_loss(output, x[:, 1, :, :])


        # Use KL divergence loss between output and x[:, 1, :, :]
        # output_log_softmax = F.log_softmax(output, dim=-1)
        # target_softmax = F.softmax(x[:, 1, :, :], dim=-1)
        # loss = F.kl_div(output_log_softmax, target_softmax, reduction="batchmean")

        std_loss = torch.abs(data_std - out_std)

        loss = mse_loss + std_loss

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.03f}",
                "out": f"{out_mean.item():.03f}, {out_std.item():.03f}",
                "mse_loss": f"{mse_loss.item():.03f}",
                "std_loss": f"{std_loss.item():.03f}",
            }
        )
    
        loss.backward()
        # for name, param in memory.named_parameters():
        #     print(name, param.grad)
        optimizer.step()
        optimizer.zero_grad()

        # for name, param in memory.current_M.named_parameters():
        #     print(name, param.requires_grad)
