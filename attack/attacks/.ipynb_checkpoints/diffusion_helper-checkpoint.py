import torch
import numpy as np

from config import DEVICE
from utils import tensor_to_numpy

class DiffusionHelper:
    def __init__(self, tokeniser, vae, scheduler, unet, text_encoder):
        self.tokeniser    = tokeniser
        self.vae          = vae
        self.scheduler    = scheduler
        self.unet         = unet
        self.text_encoder = text_encoder

    @classmethod
    def load_from_pipe(cls, pipe):
        return cls(pipe.tokenizer, pipe.vae, pipe.scheduler, pipe.unet, pipe.text_encoder)

    def _encode(self, prompt: str) -> torch.Tensor:
        tok = self.tokeniser(prompt, return_tensors="pt", padding="max_length", max_length=77).to(DEVICE)
        return self.text_encoder(**tok).last_hidden_state

    def _decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        np_image = tensor_to_numpy(image.squeeze().float().cpu().detach())
        np_image = (np_image * 255).astype(np.uint8)
        return np_image

    def _fresh_latents(self) -> torch.Tensor:
        return torch.randn(
            1, self.unet.config.in_channels,
            self.unet.config.sample_size, self.unet.config.sample_size,
        ).to(DEVICE)