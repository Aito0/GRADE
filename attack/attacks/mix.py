from config import DEVICE
from data.imagenet import select_images
from attack.attacks.normal_gen import NormalGen
from utils import numpy_to_tensor, tensor_to_numpy

import torch
from tqdm import tqdm 

class MixAttack(NormalGen):  
    @classmethod
    def load_from_pipe(cls, pipe):
        return cls(pipe.tokenizer, pipe.vae, pipe.scheduler, pipe.unet, pipe.text_encoder)
    
    @torch.no_grad()
    def generate(self, classes=1000, gens_per_class=5, time_steps=50, scaling_factor=0.5, strength=1.0, guidance_scale=7.5, dtype_=torch.float32):
        images = []
        labels = []
        for class_ in range(classes):
            for i in range(gens_per_class):
                (base_image, base_label), (conditioning_image, conditioning_label) = select_images(class_)
                
                base_tensor = numpy_to_tensor(base_image).float().unsqueeze(0).to(DEVICE)
                conditioning_tensor = numpy_to_tensor(conditioning_image).float().unsqueeze(0).to(DEVICE)
                
                cond_embeds = self.vae.encode(conditioning_tensor)
                cond_embeds = cond_embeds.latent_dist.sample() 
                cond_embeds = cond_embeds * self.vae.config.scaling_factor  # 0.18215
                text_embeds = self._encode(conditioning_label)

                uncond_embeds = self.vae.encode(base_tensor)
                uncond_embeds = uncond_embeds.latent_dist.sample() 
                uncond_embeds = uncond_embeds * self.vae.config.scaling_factor 
                
                empty_text_embeds = self._encode("")

                latents = uncond_embeds + scaling_factor * cond_embeds
                
                self.scheduler.set_timesteps(time_steps)
                start_step = int(len(self.scheduler.timesteps) * (1 - strength))
                start_timestep = self.scheduler.timesteps[start_step]

                noise = torch.randn_like(latents)
                latents = self.scheduler.add_noise(latents, noise, start_timestep.unsqueeze(0))
            
                for i, t in tqdm(enumerate(self.scheduler.timesteps[start_step:]), total=time_steps-start_step):
                    model_input = self.scheduler.scale_model_input(latents, t)

                    uncond_pred = self.unet(model_input, t, encoder_hidden_states=empty_text_embeds).sample 
                    cond_pred = self.unet(model_input, t, encoder_hidden_states=text_embeds).sample
                    
                    noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                    
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                image = self._decode_latents(latents)
            
                images.append(image)
                labels.append(base_label)

        # return base_image, base_label, conditioning_image, conditioning_label, image
        images, labels = torch.stack(images), torch.stack(labels)
        return images, labels