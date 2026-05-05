from einops import rearrange
from torchvision.transforms.functional import crop, resize
import torch


def crop_resize(
    img: torch.Tensor,
    factor: float = 0.9,
    crop_offset_x: int = 0,
    crop_offset_y: int = 0,
) -> torch.Tensor:
    if img.shape[-1] == 3 or img.shape[-1] == 6:
        if len(img.shape) == 3:
            img = rearrange(img, "h w c -> 1 c h w")
        else:
            img = rearrange(img, "b h w c -> b c h w")

    w, h = img.shape[-2], img.shape[-1]

    cropped_w = int(w * factor)
    cropped_h = int(h * factor)

    crop_top_left_x = (w - cropped_w) // 2 + crop_offset_x
    crop_top_left_y = (h - cropped_h) // 2 + crop_offset_y

    img = crop(img, crop_top_left_y, crop_top_left_x, cropped_h, cropped_w)
    img = resize(img, [w, h])
    img = rearrange(img, "b c h w -> b h w c")

    if img.shape[0] == 1:
        img = img.squeeze(0)

    return img
