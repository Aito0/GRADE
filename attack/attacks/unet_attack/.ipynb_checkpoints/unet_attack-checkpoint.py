import torch

from attack.attacks.unet_attack.unet_context import UNetContext
from attack.attacks.unet_attack.unet_attack_base import _UnetAttackBase

class UNetAttack(_UnetAttackBase):
    def blend_skips(self, ctx: UNetContext):
        return {
            k: self._blend_single_skip(ctx.skips_c[k], ctx.skips_adv[k], ctx.alpha)
            for k in ctx.skips_c
        }
    
    def _blend_single_skip(self, c_skips, adv_skips, alpha):
        if torch.is_tensor(c_skips):
            return alpha * c_skips + (1 - alpha) * adv_skips
    
        if isinstance(c_skips, tuple):
            return tuple(
                self._blend_single_skip(x, y, alpha)
                for x, y in zip(c_skips, adv_skips)
            )