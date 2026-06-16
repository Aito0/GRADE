import torch

from attack.attacks.denoise_attack.noise_context import NoiseContext
from attack.attacks.denoise_attack.denoise_attack_base import _DenoiseAttackBase

class EnhancedMixedSignal(_DenoiseAttackBase):
    def __init__(self, tokeniser, vae, scheduler, unet, text_encoder,
                 classifier=None, time_steps=50):
        super().__init__(tokeniser, vae, scheduler, unet, text_encoder, classifier)
        self.time_steps = time_steps

    @staticmethod
    def _orthogonal_component(delta_c: torch.Tensor, delta_adv: torch.Tensor) -> torch.Tensor:
        dot     = (delta_adv * delta_c).sum(dim=(1, 2, 3), keepdim=True)
        norm_sq = (delta_c   * delta_c).sum(dim=(1, 2, 3), keepdim=True)
        return delta_adv - (dot / (norm_sq + 1e-8)) * delta_c
    
    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        delta_c   = ctx.pred_c   - ctx.pred_empty
        delta_adv = ctx.pred_adv - ctx.pred_empty
        s = ctx.alpha * ((ctx.i + 1) / self.time_steps) ** ctx.beta
        return ctx.pred_empty + ctx.guidance_scale * (delta_c + s * self._orthogonal_component(delta_c, delta_adv))