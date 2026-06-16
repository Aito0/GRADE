import torch

from dataclasses import dataclass

@dataclass
class NoiseContext:
    i:              int
    t:              torch.Tensor
    model_input:    torch.Tensor
    pred_empty:     torch.Tensor
    pred_c:         torch.Tensor
    pred_adv:       torch.Tensor
    guidance_scale: float
    adversarial_t:  int
    s:              float = 1.0
    beta:           float = 1.0
    alpha:          float = 1.0