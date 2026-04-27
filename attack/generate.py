
import torch
from copy import deepcopy
from tqdm import tqdm
from utils import tensor_to_numpy, numpy_to_tensor
from config import DEVICE

def decode_image(vae, latents):
    image = vae.decode(latents / vae.config.scaling_factor).sample
    return tensor_to_numpy(image.squeeze().float().clamp(0,1).detach().cpu())

def find_grad(vae, classifier, latents, ya):
    latents = latents.detach().requires_grad_(True)

    # Decode to pixel space
    decoded = vae.decode(latents / vae.config.scaling_factor).sample
    
    # Run classifier and get log prob of target class ya
    log_prob = torch.log(classifier(decoded)[0, ya] + 1e-8)
    
    # Backprop
    log_prob.backward()
    
    return latents.grad

def generate(time_steps=50, scaling_factor=0.5, N=10, a=0.5, strength=1.0, guidance_scale=7.5, dtype_=torch.float32):    
    
    (base_image, base_label), (conditioning_image, conditioning_label) = select_images() 

    base_tensor = numpy_to_tensor(base_image).float().unsqueeze(0).to(device)
    with torch.no_grad():
        uncond_embeds = vae.encode(base_tensor)
    
        uncond_embeds = uncond_embeds.latent_dist.sample().to(device)
        
        empty_text_input = tokenizer(
            "", 
            return_tensors="pt", 
            padding='max_length', 
            max_length=64, 
            dtype=dtype_
        ).to(device)
        empty_text_embeds = text_encoder(**empty_text_input).last_hidden_state
    
        text_input = tokenizer(
            base_label, 
            return_tensors="pt", 
            padding='max_length', 
            max_length=64, 
            dtype=dtype_
        ).to(device)
        text_embeds = text_encoder(**text_input).last_hidden_state
        
        scheduler.set_timesteps(time_steps)
        start_step = int(len(scheduler.timesteps) * (1 - strength))
        start_timestep = scheduler.timesteps[start_step]
 
    latents = torch.randn_like(uncond_embeds)
    # latents = uncond_embeds
    # noise = torch.randn_like(uncond_embeds)
    # latents = scheduler.add_noise(uncond_embeds, noise, start_timestep.unsqueeze(0))
    x_T = deepcopy(latents)
        
    for i in range(N):
        latents = deepcopy(x_T)
        for j, t in tqdm(enumerate(scheduler.timesteps[start_step:]), total=time_steps-start_step):
            model_input = scheduler.scale_model_input(latents, t)

            with torch.no_grad():
                uncond_pred = unet(model_input, t, encoder_hidden_states=empty_text_embeds).sample 
                cond_pred = unet(model_input, t, encoder_hidden_states=text_embeds).sample
                
                noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            img = _decode_image(latents)
            
        mispred_prob = evaluate(img)
        print(f"step {i}: {mispred_prob}")
        # x_T += (1 - scheduler.alphas_cumprod[-1]).item() * a * (_find_grad(latents, label_to_idx[conditioning_label]))

    with torch.no_grad():
        # Decode image
        image = _decode_image(latents)

    return base_image, base_label, conditioning_image, conditioning_label, image

base_image, base_label, conditioning_image, conditioning_label, image = generate(
    time_steps=50, 
    # scaling_factor=0.4,
    strength=1.0,
    N=1,
    # a=0.5,
    guidance_scale=7.5
)

def display(image):
    if not isinstance(image, np.ndarray):
        image = image.float().numpy()
    
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def evaluate(image, model=swin_b):
    pred_image = numpy_to_tensor(image).unsqueeze(0)
    preds = model(pred_image.to(device))
    pred_logits = torch.nn.functional.softmax(preds[0], dim=0)
    pred_class = torch.argmax(pred_logits).item()
    return idx_to_label[pred_class], pred_logits[pred_class].item()

if __name__ == "__main__":
    main()