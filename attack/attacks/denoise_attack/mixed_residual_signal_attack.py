import torch

from attack.attacks.denoise_attack.noise_context import NoiseContext
from attack.attacks.denoise_attack.denoise_attack_base import _DenoiseAttackBase

class MixedResidualSignalAttack(_DenoiseAttackBase):
    """Mix the full adversarial residual (delta_adv - delta_c) into guidance."""

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        delta_c = ctx.pred_c - ctx.pred_empty
        delta_adv = ctx.pred_adv - ctx.pred_empty
        return ctx.pred_empty + ctx.guidance_scale * (delta_c + ctx.s * (delta_adv - delta_c))