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
        print("updating memory...")
        print("new_value shape:", new_value.shape)
        B, L, D = new_value.shape
        if not hasattr(self, "memory"):
            self.memory = torch.zeros(self.num_slots, B, L, D, device=new_value.device)  # Initialize memory

        # Update the memory with the new value
        self.value = new_value
        self.memory = torch.roll(self.memory, shifts=-1)  # Shift memory to the left
        self.memory[-1] = new_value  # Add new value to the end of memory

    def retrieve(self, new_value: torch.Tensor) -> torch.Tensor:
        print("retrieving from memory...")
        print("new_value shape:", new_value.shape)
        B, L, D = new_value.shape
        if not hasattr(self, "memory"):
            self.memory = torch.zeros(self.num_slots, B, L, D, device=new_value.device)  # Initialize memory

        # Retrieve the current value from memory
        memory_rearranged = rearrange(
            self.memory, "n b l d -> b (n l) d"
        )  # Rearrange memory for attention

        print("new_value shape:", new_value.shape)
        print("memory_rearranged shape:", memory_rearranged.shape)

        memory_rearranged = memory_rearranged.detach()  # Detach memory to prevent gradients from flowing back
        attn_output, _ = self.attn(new_value, memory_rearranged, memory_rearranged)
        return attn_output

    def reset_memory(self):
        if hasattr(self, "memory"):
            del self.memory  # Clear memory when resetting


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
