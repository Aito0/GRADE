import torch

from attack.attacks.denoise_attack.noise_context import NoiseContext
from attack.attacks.denoise_attack.denoise_attack_base import _DenoiseAttackBase

class MixedTimedResidualSignalAttack(_DenoiseAttackBase):
    """Scale adversarial residual by a power-law schedule s(i) = ((i+1)/T)^beta."""

    def __init__(self, tokeniser, vae, scheduler, unet, text_encoder,
                 classifier=None, time_steps=50):
        super().__init__(tokeniser, vae, scheduler, unet, text_encoder, classifier)
        self.time_steps = time_steps
        
    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        delta_c   = ctx.pred_c   - ctx.pred_empty
        delta_adv = ctx.pred_adv - ctx.pred_empty
        s = ((ctx.i + 1) / self.time_steps) ** ctx.beta
        return ctx.pred_empty + ctx.guidance_scale * (delta_c + s * (delta_adv - delta_c))