import torch
import numpy as np

from tqdm import tqdm
from abc import ABC, abstractmethod

from attack.attacks.base_attack import _BaseAttack
from attack.attacks.denoise_attack.noise_context import NoiseContext

class _DenoiseAttackBase(_BaseAttack, ABC):
    def __init__(
        self,
        tokeniser,
        vae,
        scheduler,
        unet,
        text_encoder,
        classifier=None,
    ):
        super().__init__(
            tokeniser,
            vae,
            scheduler,
            unet,
            text_encoder,
            classifier,
        )

    @classmethod
    def load_from_pipe(cls, pipe, **kwargs):
        return cls(pipe.tokenizer, pipe.vae, pipe.scheduler, pipe.unet, pipe.text_encoder, **kwargs)

    @abstractmethod
    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def attack(self, time_steps, guidance_scale, true_class, fake_class, latents, adversarial_t, beta=1.0, s=1.0, alpha=1.0) -> np.ndarray:
        self.scheduler.set_timesteps(time_steps)

        empty_emb = self._encode("")
        c_emb     = self._encode(true_class)
        adv_emb   = self._encode(fake_class)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            model_input = self.scheduler.scale_model_input(latents, t)
            
            pred_empty = self.unet(model_input, t, encoder_hidden_states=empty_emb).sample
            pred_c     = self.unet(model_input, t, encoder_hidden_states=c_emb).sample
            pred_adv   = self.unet(model_input, t, encoder_hidden_states=adv_emb).sample
            
            if i > adversarial_t:
                ctx = NoiseContext(
                    i=i, t=t,
                    model_input    = model_input,
                    pred_empty     = pred_empty,
                    pred_c         = pred_c,
                    pred_adv       = pred_adv,
                    guidance_scale = guidance_scale,
                    adversarial_t  = adversarial_t,
                    beta           = beta,
                    s              = s,
                    alpha          = alpha,
                )
                noise_pred = self._compute_noise(ctx)
            else:
                noise_pred = pred_empty + guidance_scale * (pred_c - pred_empty)
                
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return self._decode_latents(latents)

    def run_attack(
        self,
        true_class:     str,
        time_steps:     int   = 50,
        guidance_scale: float = 10,
        latents:        torch.Tensor | None = None,
        fake_class:     str | None = None,
        s:              float = 0.3,
        adversarial_t:  int = 20,
        beta:           float = 1.0,
        alpha:          float = 1.0
    ) -> tuple[np.ndarray, str, np.ndarray]:
        
        latents, clean_image, fake_class = self.generate_verified(
            true_class, time_steps, guidance_scale, latents, fake_class)

        if latents is None:
            return None, None, None
        
        attacked_image = self._run_attack(time_steps, guidance_scale, true_class, fake_class, latents, adversarial_t=adversarial_t, s=s, beta=beta, alpha=alpha)
        return attacked_image, fake_class, clean_image