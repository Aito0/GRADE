import torch

from dataclasses import dataclass

@dataclass
class UNetContext:
    i:              int
    t:              torch.Tensor
    timesteps:      int
    model_input:    torch.Tensor
    skips_empty:    torch.Tensor
    skips_c:        torch.Tensor
    skips_adv:      torch.Tensor
    guidance_scale: float
    alpha:          float