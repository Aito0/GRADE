from diffusers import StableDiffusionPipeline, DDIMScheduler
from config import DEVICE, STABLE_DIFFUSION_DIR

def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        STABLE_DIFFUSION_DIR, local_files_only=True
    ).to(DEVICE)
    return pipe

def load_scheduler():
    return DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )