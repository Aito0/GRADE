# main.py
from config import SEED, DEVICE, SAVE_DIR, set_seed
from models.classifiers import load_classifier
from models.diffusion import load_pipeline
from attack.normal_gen import MixedSignalAttack
from evaluation.evaluate import evaluate, HyperparameterEvaluator

import torch
import numpy as np
from diffusers import DDIMScheduler, DiffusionPipeline

def main():
    set_seed(SEED)

    pipe = DiffusionPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-1-base",
        torch_dtype=torch.float32,
    ).to(DEVICE)

    pipe.scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    # load precomputed latents and fake classes
    data    = np.load("test_imagenet_attack_latents.npz", allow_pickle=True)
    latents = torch.from_numpy(data["latents"]).to(DEVICE)   # (N, C, H, W)
    
    labels          = data["labels"].tolist()                         # true class per latent
    fake_classes    = data["fake_classes"].tolist()                 # fake class per latent
    guidance_scales = data["guidance_scales"].tolist()

    classes = [
        "golden retriever",
        "fire truck",
        # "spee",
        "church",
    ]

    hyperparameter_grid = {
        "guidance_scale": [7.5],#[5, 7.5, 10, 12.5, 15],
        "s":              [0.4],#[0.2, 0.4, 0.6, 0.8],
        # "time_steps":     [1][15, 25, 35],
    }

    classifiers = {name: load_classifier(name) for name in [
        "RES50", "VGG19", "MOB_V2", "INC_V3", "CONVNEXT", 
        "SWIN_B", "DEIT_B", "DEIT_S", "MIX_B", "MIX_L",
    ]}

    evaluator = HyperparameterEvaluator(
        attack_class=MixedSignalAttack,
        pipe=pipe,
        classifiers=classifiers,
        evaluate_image_fn=evaluate,
        write_path="results/",
        precomputed_latents=latents,
        precomputed_labels=labels,
        precomputed_fake_classes=fake_classes,
        precomputed_guidance_scales=guidance_scales,
    )

    evaluator.run(
        classes=classes,
        hyperparameter_grid=hyperparameter_grid,
        output_csv="results.csv",
    )

if __name__ == "__main__":
    main()