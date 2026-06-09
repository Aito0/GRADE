from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm
import logging

from config import DEVICE
from data.imagenet import idx_to_label 
from models.classifiers import load_classifier
from utils import tensor_to_numpy, numpy_to_tensor


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
        return tensor_to_numpy(image.squeeze().float().cpu().detach())

    def _fresh_latents(self) -> torch.Tensor:
        print("Creating fresh latents")
        return torch.randn(
            1, self.unet.config.in_channels,
            self.unet.config.sample_size, self.unet.config.sample_size,
        ).to(DEVICE)


class NormalGen(DiffusionHelper):

    def generate(self, prompt, time_steps=50, guidance_scale=7.5, latents=None):
        self.scheduler.set_timesteps(time_steps)

        text_embeds  = self._encode(prompt)
        empty_embeds = self._encode("")
        latents      = latents if latents is not None else self._fresh_latents()

        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps):
                model_input            = self.scheduler.scale_model_input(latents, t)
                noise_pred             = self.unet(torch.cat([model_input] * 2), t,
                                                   encoder_hidden_states=torch.cat([empty_embeds, text_embeds])).sample
                uncond_pred, cond_pred = noise_pred.chunk(2)
                noise_pred             = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                latents                = self.scheduler.step(noise_pred, t, latents).prev_sample

        return self._decode_latents(latents)


# ---------------------------------------------------------------------------
# Attack classes
# ---------------------------------------------------------------------------

@dataclass
class NoiseContext:
    i:              int
    t:              torch.Tensor
    model_input:    torch.Tensor
    pred_empty:     torch.Tensor
    pred_c:         torch.Tensor
    pred_adv:       torch.Tensor
    guidance_scale: float


class _DenoiseAttackBase(NormalGen):

    def __init__(self, tokeniser, vae, scheduler, unet, text_encoder,
                 classifier=None, adversarial_timesteps=np.arange(20, 51), s=1):
        super().__init__(tokeniser, vae, scheduler, unet, text_encoder)
        self.classifier            = classifier.to(DEVICE) if classifier is not None else load_classifier("SWIN_B").to(DEVICE)
        self.adversarial_timesteps = adversarial_timesteps
        self.s                     = s                 = s

    @classmethod
    def load_from_pipe(cls, pipe, **kwargs):
        return cls(pipe.tokenizer, pipe.vae, pipe.scheduler, pipe.unet, pipe.text_encoder, **kwargs)

    @torch.no_grad()
    def get_top2_classes(self, image: np.ndarray) -> tuple[str, str]:
        logits = self.classifier(numpy_to_tensor(image).unsqueeze(0).to(DEVICE))
        top2   = torch.softmax(logits, dim=-1).topk(2).indices.squeeze()
        return idx_to_label(top2[0].item()), idx_to_label(top2[1].item())

    def generate_verified(
        self,
        true_class:     str,
        time_steps:     int,
        guidance_scale: float,
        latents:        torch.Tensor | None = None,
        fake_class:     str | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, str]:
        if fake_class is not None:
            latents = latents if latents is not None else self._fresh_latents()
            return latents, self.generate(true_class, time_steps, guidance_scale, latents), fake_class

        top_class = ""
        while top_class != true_class:
            latents     = self._fresh_latents()
            clean_image = self.generate(true_class, time_steps, guidance_scale, latents)
            top_class, fake_class = self.get_top2_classes(clean_image)

        return latents, clean_image, fake_class

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def _run_attack(self, time_steps, guidance_scale, true_class, fake_class, latents) -> np.ndarray:
        self.scheduler.set_timesteps(time_steps)

        empty_emb = self._encode("")
        c_emb     = self._encode(true_class)
        adv_emb   = self._encode(fake_class)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            model_input = self.scheduler.scale_model_input(latents, t)
            ctx = NoiseContext(
                i=i, t=t,
                model_input=model_input,
                pred_empty=self.unet(model_input, t, encoder_hidden_states=empty_emb).sample,
                pred_c    =self.unet(model_input, t, encoder_hidden_states=c_emb).sample,
                pred_adv  =self.unet(model_input, t, encoder_hidden_states=adv_emb).sample,
                guidance_scale=guidance_scale,
            )
            latents = self.scheduler.step(self._compute_noise(ctx), t, latents).prev_sample

        return self._decode_latents(latents)

    def run_attack(
        self,
        true_class:     str,
        time_steps:     int   = 50,
        guidance_scale: float = 10,
        latents:        torch.Tensor | None = None,
        fake_class:     str | None = None,
        s:              float = None,
        adversarial_timesteps = None,
    ) -> tuple[np.ndarray, str, np.ndarray]:
        if s is not None:
            self.s = s
        if adversarial_timesteps is not None:
            self.adversarial_timesteps = adversarial_timesteps

        latents, clean_image, fake_class = self.generate_verified(
            true_class, time_steps, guidance_scale, latents, fake_class)
        attacked_image = self._run_attack(time_steps, guidance_scale, true_class, fake_class, latents)
        return attacked_image, fake_class, clean_image


class MixedSignalAttack(_DenoiseAttackBase):
    """Replace full guidance with adversarial guidance at selected timesteps."""

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        noise_c = ctx.pred_empty + ctx.guidance_scale * (ctx.pred_c   - ctx.pred_empty)
        if ctx.i not in self.adversarial_timesteps:
            return noise_c
        noise_adv = ctx.pred_empty + ctx.guidance_scale * (ctx.pred_adv - ctx.pred_empty)
        return noise_adv + self.s * (noise_c - noise_adv)


class MixedResidualSignalAttack(_DenoiseAttackBase):
    """Mix the full adversarial residual (delta_adv − delta_c) into guidance."""

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        delta_c = ctx.pred_c - ctx.pred_empty
        if ctx.i not in self.adversarial_timesteps:
            return ctx.pred_empty + ctx.guidance_scale * delta_c
        delta_adv = ctx.pred_adv - ctx.pred_empty
        return ctx.pred_empty + ctx.guidance_scale * (delta_c + self.s * (delta_adv - delta_c))


class MixedNormalResidualSignalAttack(_DenoiseAttackBase):
    """Add only the component of delta_adv orthogonal to delta_c."""

    @staticmethod
    def _orthogonal_component(delta_c: torch.Tensor, delta_adv: torch.Tensor) -> torch.Tensor:
        dot     = (delta_adv * delta_c).sum(dim=(1, 2, 3), keepdim=True)
        norm_sq = (delta_c   * delta_c).sum(dim=(1, 2, 3), keepdim=True)
        return delta_adv - (dot / (norm_sq + 1e-8)) * delta_c

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        delta_c = ctx.pred_c - ctx.pred_empty
        if ctx.i not in self.adversarial_timesteps:
            return ctx.pred_empty + ctx.guidance_scale * delta_c
        delta_adv = ctx.pred_adv - ctx.pred_empty
        return ctx.pred_empty + ctx.guidance_scale * (delta_c + self.s * self._orthogonal_component(delta_c, delta_adv))


class MixedTimedResidualSignalAttack(_DenoiseAttackBase):
    """Scale adversarial residual by a power-law schedule s(i) = ((i+1)/T)^beta."""

    def __init__(self, tokeniser, vae, scheduler, unet, text_encoder,
                 classifier=None, time_steps=50, beta=2):
        super().__init__(tokeniser, vae, scheduler, unet, text_encoder, classifier)
        self.time_steps = time_steps
        self.beta       = beta

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        delta_c   = ctx.pred_c   - ctx.pred_empty
        delta_adv = ctx.pred_adv - ctx.pred_empty
        s = ((ctx.i + 1) / self.time_steps) ** self.beta
        return ctx.pred_empty + ctx.guidance_scale * (delta_c + s * (delta_adv - delta_c))