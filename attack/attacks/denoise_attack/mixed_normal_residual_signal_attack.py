import torch

from attack.attacks.denoise_attack.noise_context import NoiseContext
from attack.attacks.denoise_attack.denoise_attack_base import _DenoiseAttackBase

class MixedNormalResidualSignalAttack(_DenoiseAttackBase):
    """Add only the component of delta_adv orthogonal to delta_c."""

    @staticmethod
    def _orthogonal_component(delta_c: torch.Tensor, delta_adv: torch.Tensor) -> torch.Tensor:
        dot     = (delta_adv * delta_c).sum(dim=(1, 2, 3), keepdim=True)
        norm_sq = (delta_c   * delta_c).sum(dim=(1, 2, 3), keepdim=True)
        return delta_adv - (dot / (norm_sq + 1e-8)) * delta_c

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        delta_c = ctx.pred_c - ctx.pred_empty
        delta_adv = ctx.pred_adv - ctx.pred_empty
        return ctx.pred_empty + ctx.guidance_scale * (delta_c + ctx.s * self._orthogonal_component(delta_c, delta_adv))