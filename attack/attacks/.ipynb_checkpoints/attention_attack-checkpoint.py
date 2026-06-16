import torch

from attack.attacks.base_attack import _BaseAttack

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
                    text_embeds_c + alpha * (text_embeds_adv - text_embeds_c) 
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
        adversarial_t,
        blocks_changed,
        alpha
    ):
        self.scheduler.set_timesteps(time_steps)

        empty_emb = self._encode("")
        c_emb     = self._encode(true_class)
        adv_emb   = self._encode(fake_class)

        for i, t in enumerate(self.scheduler.timesteps):
            model_input = self.scheduler.scale_model_input(latents, t)
            
            pred_empty = self.unet(model_input,t, encoder_hidden_states=empty_emb).sample
            if i > adversarial_t:
                pred_c = self._adv_unet(model_input, t, c_emb, adv_emb, blocks_changed, alpha)
            else:
                pred_c = self.unet(model_input,t,encoder_hidden_states=c_emb).sample

            noise_pred = pred_empty + guidance_scale * (pred_c - pred_empty)
            latents    = self.scheduler.step(noise_pred,t,latents,).prev_sample

        return self._decode_latents(latents)

    def attack(
        self,
        true_class,
        time_steps=50,
        guidance_scale=7.5,
        latents=None,
        fake_class=None,
        adversarial_t=20,
        blocks_changed=2,
        alpha=0.3,
    ):
        
        latents, clean_image, fake_class = self.generate_verified(true_class,time_steps,guidance_scale,latents,fake_class)

        if latents is None:
            return None, None, None
        
        attacked_image = self._run_attack(time_steps,guidance_scale,true_class,fake_class,latents,adversarial_t,blocks_changed,alpha)
        return attacked_image, fake_class, clean_image