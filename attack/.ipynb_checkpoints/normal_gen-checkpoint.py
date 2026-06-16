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
        np_image = tensor_to_numpy(image.squeeze().float().cpu().detach())
        np_image = (np_image * 255).astype(np.uint8)
        return np_image

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

class _BaseAttack(NormalGen):
    def __init__(
        self,
        tokeniser,
        vae,
        scheduler,
        unet,
        text_encoder,
        classifier=None,
    ):
        super().__init__(tokeniser, vae, scheduler, unet, text_encoder)
        self.classifier = (
            classifier.to(DEVICE)
            if classifier is not None
            else load_classifier("SWIN_B")
        )

    @torch.no_grad()
    def get_top2_classes(self, image: np.ndarray) -> tuple[str, str]:
        logits = self.classifier(numpy_to_tensor(image).unsqueeze(0).to(DEVICE))
        top2 = torch.softmax(logits, dim=-1).topk(2).indices.squeeze()
        return idx_to_label(top2[0].item()), idx_to_label(top2[1].item())

    def generate_verified(
        self,
        true_class: str,
        time_steps: int,
        guidance_scale: float,
        latents: torch.Tensor | None = None,
        fake_class: str | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, str]:

        if fake_class is not None:
            latents = latents if latents is not None else self._fresh_latents()
            return latents, self.generate(true_class,time_steps,guidance_scale,latents), fake_class

        print("Recalculating latents")
        top_class = ""
        for i in range(5):
            latents = self._fresh_latents()
            clean_image = self.generate(true_class,time_steps,guidance_scale,latents)
            top_class, fake_class = self.get_top2_classes(clean_image)

            if top_class == true_class:
                break

        if top_class != true_class:
            return None, None, None
        
        return latents, clean_image, fake_class

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


class _DenoiseAttackBase(_BaseAttack):
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

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def _run_attack(self, time_steps, guidance_scale, true_class, fake_class, latents, adversarial_t, beta=1.0, s=1.0, alpha=1.0) -> np.ndarray:
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


class MixedSignalAttack(_DenoiseAttackBase):
    """Replace full guidance with adversarial guidance at selected timesteps."""

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        noise_c = ctx.pred_empty + ctx.guidance_scale * (ctx.pred_c   - ctx.pred_empty)        
        noise_adv = ctx.pred_empty + ctx.guidance_scale * (ctx.pred_adv - ctx.pred_empty)
        return noise_c + ctx.s * (noise_adv - noise_c)


class MixedResidualSignalAttack(_DenoiseAttackBase):
    """Mix the full adversarial residual (delta_adv − delta_c) into guidance."""

    def _compute_noise(self, ctx: NoiseContext) -> torch.Tensor:
        delta_c = ctx.pred_c - ctx.pred_empty
        delta_adv = ctx.pred_adv - ctx.pred_empty
        return ctx.pred_empty + ctx.guidance_scale * (delta_c + ctx.s * (delta_adv - delta_c))


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

class NormalisedEnhancedMixedSignal(_DenoiseAttackBase):
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

        combined = delta_c + s * self._orthogonal_component(delta_c, delta_adv)
        scale_c = torch.linalg.vector_norm(delta_c, dim=(1, 2, 3), keepdim=True)
        scale_combined = torch.linalg.vector_norm(combined, dim=(1, 2, 3), keepdim=True)
        scale = scale_c / (scale_combined + 1e-8)
        combined = combined * scale
        return ctx.pred_empty + ctx.guidance_scale * (combined)

@dataclass
class UNetContext:
    i:              int
    t:              torch.Tensor
    timesteps:      int
    model_input:    torch.Tensor
    skips_empty:    torch.Tensor
    skips_c:        torch.Tensor
    skips_adv:      torch.Tensor
    guidance_scale: float
    alpha:          float

class _UnetAttackBase(_BaseAttack):
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

    def clear_hooks(self):
        for module in self.unet.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
            module._backward_hooks.clear()

    def get_down_block_map(self, replace_blocks):
        down_blocks = [self.unet.conv_in] + list(self.unet.down_blocks) + [self.unet.mid_block]
        down_block_names = ["conv_in"] + [f"block_{i+1}" for i in range(len(self.unet.down_blocks))] + ["mid_block"]
        down_block_map = {block: name for block, name in zip(down_blocks[:replace_blocks+1], down_block_names[:replace_blocks+1])}
        return down_block_map
        
    def get_up_block_map(self, replace_blocks):
        up_blocks = list(self.unet.up_blocks)[::-1] + [self.unet.mid_block]
        up_block_names = [f"block_{i+1}" for i in range(len(list(self.unet.up_blocks)[::-1]))] + ["mid_block"]
        up_block_map = {block: name for block, name in zip(up_blocks[:replace_blocks], up_block_names[:replace_blocks])}
        return up_block_map
    
    def get_early_features(self, unet_input, t, encoder_hidden_states, replace_blocks):
        self.clear_hooks()
        skip_connections = {}
        def get_hook_down(name):
            def store_latents(mod, args, kwargs, out):
                # print(f"{name} skips: {len(skips)} tensors, shapes: {[s.shape for s in skips]}")
                skip_connections[name] = out[1] if isinstance(out, tuple) else out
            return store_latents

        down_blocks_map = self.get_down_block_map(replace_blocks)
        down_handlers = [
            block.register_forward_hook(get_hook_down(name), with_kwargs=True) 
            for block, name in down_blocks_map.items()
        ]
        with torch.no_grad():
            new_latents = self.unet(unet_input, t, encoder_hidden_states=encoder_hidden_states).sample
        
        for down_handler in down_handlers:
            down_handler.remove()
        
        return new_latents, skip_connections

    def get_skip_stack(self, skips):
        ordered_keys = ["conv_in","block_1","block_2","block_3","block_4"]
        skip_stack = []
        for key in ordered_keys:
            if key not in skips:
                continue

            v = skips[key]        
            if torch.is_tensor(v):
                skip_stack.append(v)
            else:
                skip_stack.extend(v)

        return skip_stack
    
    def texture_unet(self, unet_input, t, encoder_hidden_states, skips, replace_blocks):
        def get_hook_up(name):
            def inject_skip_connections(mod, args, kwargs):
                try:
                    incoming = kwargs['res_hidden_states_tuple']  
                    incoming = tuple(
                        skip_stack.pop(-1)
                        for _ in incoming
                    )[::-1]
                    kwargs["res_hidden_states_tuple"] = incoming
                    return args, kwargs
                except Exception as e:
                    print("Error:", e)
            return inject_skip_connections

        skip_stack = self.get_skip_stack(skips)
        if replace_blocks != len(self.unet.down_blocks):
            skip_stack.pop(-1)
            
        up_blocks_map = self.get_up_block_map(replace_blocks)
        up_handlers = [
            block.register_forward_pre_hook(get_hook_up(name), with_kwargs=True) 
            for block, name in up_blocks_map.items()
        ]
        with torch.no_grad():
            out = self.unet(unet_input, t, encoder_hidden_states=encoder_hidden_states)

        for up_handler in up_handlers:
            up_handler.remove()
            
        return out.sample
    
    def blend_skips(self, ctx):
        raise NotImplementedError

    def _run_attack(self, time_steps, guidance_scale, true_class, fake_class, latents, adversarial_t, alpha, blocks_replaced):        
        self.scheduler.set_timesteps(time_steps)
        
        text_embeds_c   = self._encode(true_class)
        text_embeds_adv = self._encode(fake_class) 
        empty_embeds    = self._encode("")
        
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=time_steps):
            model_input = self.scheduler.scale_model_input(latents, t)
            with torch.no_grad():
                
                if i > adversarial_t:
                    noise_c, skips_c   = self.get_early_features(model_input, t, text_embeds_c, blocks_replaced)
                    noise_adv, skips_adv = self.get_early_features(model_input, t, text_embeds_adv, blocks_replaced)
                    uncond_pred, empty_skips = self.get_early_features(model_input, t, empty_embeds, blocks_replaced)

                    ctx = UNetContext(
                        i=i,t=t,
                        timesteps=time_steps,
                        model_input=model_input,
                        skips_empty=empty_skips,
                        skips_c=skips_c,
                        skips_adv=skips_adv,
                        guidance_scale=guidance_scale,
                        alpha=alpha
                    )
                    skips = self.blend_skips(ctx)
                    cond_pred = self.texture_unet(model_input, t, text_embeds_c, skips, blocks_replaced)
                    skips_c.clear()
                    skips_adv.clear()
                    empty_skips.clear()
                else:
                    uncond_pred = self.unet(model_input, t, encoder_hidden_states=empty_embeds).sample
                    cond_pred   = self.unet(model_input, t, encoder_hidden_states=text_embeds_c).sample
                    
                noise_pred_c = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                latents      = self.scheduler.step(noise_pred_c, t, latents).prev_sample
    
        image = self._decode_latents(latents)
        return image
    
    def run_attack(
        self,
        true_class:      str,
        time_steps:      int   = 50,
        guidance_scale:  float = 10,
        latents:         torch.Tensor | None = None,
        fake_class:      str | None = None,
        adversarial_t:  int = 20,
        alpha:           int = 0.7,
        blocks_replaced: int = 2,
    ) -> tuple[np.ndarray, str, np.ndarray]:

        latents, clean_image, fake_class = self.generate_verified(
            true_class, time_steps, guidance_scale, latents, fake_class)

        if latents is None:
            return None, None, None
        
        attacked_image = self._run_attack(time_steps, guidance_scale, true_class, fake_class, latents, adversarial_t, alpha, blocks_replaced)
        return attacked_image, fake_class, clean_image

class BaseUNetGen(_UnetAttackBase):
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

class AttentionAttack(_BaseAttack):
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
        return cls(
            pipe.tokenizer,
            pipe.vae,
            pipe.scheduler,
            pipe.unet,
            pipe.text_encoder,
            **kwargs,
        )

    def clear_hooks(self):
        for module in self.unet.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
            module._backward_hooks.clear()

    def _adv_unet(
        self,
        model_input,
        t,
        text_embeds_c,
        text_embeds_adv,
        blocks_changed,
        alpha
    ):
        self.clear_hooks()

        def cross_attention_hook(name):
            def replace_attention(mod, args, kwargs):
                kwargs["encoder_hidden_states"] = (
                    alpha * text_embeds_c + (1-alpha) * text_embeds_adv
                    if name in change_blocks
                    else text_embeds_c
                )
                return args, kwargs
            return replace_attention

        depth = len(self.unet.down_blocks)

        block_map = {self.unet.mid_block: self.unet.mid_block}
        block_map.update({
            self.unet.down_blocks[depth - 1 - i]:
            self.unet.up_blocks[i]
            for i in range(depth)
        })

        blocks = [self.unet.mid_block]
        blocks.extend(self.unet.down_blocks)

        change_blocks = {
            f"block_{i}"
            for i in range(blocks_changed + 1)
        }

        handlers = []

        for i, block in enumerate(blocks):

            handlers.append(
                block.register_forward_pre_hook(
                    cross_attention_hook(f"block_{i}"),
                    with_kwargs=True,
                )
            )

            if block != self.unet.mid_block:
                handlers.append(
                    block_map[block].register_forward_pre_hook(
                        cross_attention_hook(f"block_{i}"),
                        with_kwargs=True,
                    )
                )

        pred = self.unet(model_input,t,encoder_hidden_states=text_embeds_c).sample

        for h in handlers:
            h.remove()

        return pred

    @torch.no_grad()
    def _run_attack(
        self,
        time_steps,
        guidance_scale,
        true_class,
        fake_class,
        latents,
        blocks_changed,
        adversarial_t,
        alpha,
    ):
        self.scheduler.set_timesteps(time_steps)

        empty_emb = self._encode("")
        c_emb     = self._encode(true_class)
        adv_emb   = self._encode(fake_class)

        for i, t in enumerate(self.scheduler.timesteps):
            model_input = self.scheduler.scale_model_input(latents, t)
            
            pred_empty = self.unet(model_input,t, encoder_hidden_states=empty_emb).sample
            if i > adversarial_t:
                pred_c = self._adv_unet(model_input,t,c_emb,adv_emb,blocks_changed, alpha)
            else:
                pred_c = self.unet(model_input,t,encoder_hidden_states=c_emb).sample

            noise_pred = pred_empty + guidance_scale * (pred_c - pred_empty)
            latents    = self.scheduler.step(noise_pred,t,latents,).prev_sample

        return self._decode_latents(latents)

    def run_attack(
        self,
        true_class,
        time_steps=50,
        guidance_scale=7.5,
        latents=None,
        fake_class=None,
        blocks_changed=2,
        adversarial_t=20,
        alpha=1.0,
    ):
        
        latents, clean_image, fake_class = self.generate_verified(true_class,time_steps,guidance_scale,latents,fake_class)

        if latents is None:
            return None, None, None
        
        attacked_image = self._run_attack(time_steps,guidance_scale,true_class,fake_class,latents,blocks_changed,adversarial_t, alpha)
        return attacked_image, fake_class, clean_image