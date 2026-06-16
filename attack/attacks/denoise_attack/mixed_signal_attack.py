import torch

from attack.attacks.denoise_attack.noise_context import NoiseContext
from attack.attacks.denoise_attack.denoise_attack_base import _DenoiseAttackBase

class MixedSignalAttack(_DenoiseAttackBase):
    """Replace full guidance with adversarial guidance at selected timesteps."""

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        noise_c = ctx.pred_empty + ctx.guidance_scale * (ctx.pred_c   - ctx.pred_empty)        
        noise_adv = ctx.pred_empty + ctx.guidance_scale * (ctx.pred_adv - ctx.pred_empty)
        return noise_c + ctx.s * (noise_adv - noise_c)
