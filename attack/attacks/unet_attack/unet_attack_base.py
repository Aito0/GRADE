import torch
import numpy as np

from tqdm import tqdm
from abc import ABC, abstractmethod

from attack.attacks.base_attack import _BaseAttack
from attack.attacks.unet_attack.unet_context import UNetContext

class _UnetAttackBase(_BaseAttack, ABC):
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
    
    @abstractmethod
    def blend_skips(self, ctx: UNetContext):
        pass

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
