import torch
from torch import nn
from einops import rearrange


class SlotMemory(nn.Module):
    def __init__(self, embed_dim: int, num_slots: int = 10):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=12, batch_first=True
        )  # Example attention layer
        self.embed_dim = embed_dim
        self.num_slots = num_slots

    def update(self, new_value: torch.Tensor):
        B, L, D = new_value.shape
        if not hasattr(self, "memory"):
            self.memory = torch.zeros(
                B, self.num_slots, L, D, device=new_value.device
            )  # Initialize memory

        # Update the memory with the new value
        self.value = new_value
        self.memory = torch.roll(
            self.memory, shifts=-1, dims=1
        )  # Shift memory to the left
        self.memory[:, -1] = new_value  # Add new value to the end of memory

    def retrieve(self, new_value: torch.Tensor) -> torch.Tensor:
        B, L, D = new_value.shape
        if not hasattr(self, "memory"):
            self.memory = torch.zeros(
                B, self.num_slots, L, D, device=new_value.device
            )  # Initialize memory

        # Retrieve the current value from memory
        memory_rearranged = rearrange(
            self.memory, "b n l d -> b (n l) d"
        )  # Rearrange memory for attention

        memory_rearranged = (
            memory_rearranged.detach()
        )  # Detach memory to prevent gradients from flowing back
        attn_output, _ = self.attn(new_value, memory_rearranged, memory_rearranged)
        return attn_output

    def reset_memory(self, mask: torch.Tensor | None = None):
        if hasattr(self, "memory"):
            if mask is None or mask.shape[0] != self.memory.shape[0]:
                del self.memory  # Clear memory when resetting
            else:
                mask = rearrange(mask, "b -> b 1 1 1")  # [B, 1, 1, 1]
                self.memory = torch.where(
                    mask, torch.zeros_like(self.memory), self.memory
                )  # Set memory to zero where mask is true


if __name__ == "__main__":
    batch_size = 5
    memory_size = torch.Size((10, 256, 768))  # Example memory size
    slot_memory = SlotMemory(768)

    # Example of updating memory with new values
    for i in range(5):
        new_value = torch.randn(
            batch_size, 256, 768
        )  # Simulate new value as a random tensor
        slot_memory.update(new_value)
        print(f"Updated memory with new value: {new_value.shape}")

    # Example of retrieving value from memory
    query_value = torch.randn(
        batch_size, 256, 768
    )  # Simulate query value as a random tensor
    retrieved_value = slot_memory.retrieve(query_value)
    print(f"Retrieved value from memory: {retrieved_value.shape}")
