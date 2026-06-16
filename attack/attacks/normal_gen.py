import torch

from tqdm import tqdm

from attack.attacks.diffusion_helper import DiffusionHelper

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
