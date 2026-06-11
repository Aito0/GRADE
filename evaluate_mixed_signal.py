# main.py
from config import SEED, DEVICE, SAVE_DIR, set_seed
from models.classifiers import load_classifier
from models.diffusion import load_pipeline
from attack.normal_gen import MixedSignalAttack, MixedResidualSignalAttack, MixedNormalResidualSignalAttack, MixedTimedResidualSignalAttack
from evaluation.evaluate import evaluate, HyperparameterEvaluator

import torch
import numpy as np
from diffusers import DDIMScheduler, DiffusionPipeline

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mixed_signal.log"),
    ]
)
log = logging.getLogger(__name__)

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
    data    = np.load("imagenet_attack_latents_combined.npz", allow_pickle=True)
    latents = torch.from_numpy(data["latents"])
    latents = latents.to(DEVICE)   # (N, C, H, W)
    
    labels          = data["labels"].tolist()                         # true class per latent
    fake_classes    = data["fake_classes"].tolist()                 # fake class per latent
    guidance_scales = data["guidance_scales"].tolist()

    classes = [
        "golden retriever",
        "fire engine",
        "speedboat",
        "church",
    ]

    hyperparameter_grid = {
        "guidance_scale": [5, 7.5, 10, 12.5, 15],
        "s":              [0.2, 0.4, 0.6, 0.8],
        "adversarial_t":  [15, 25, 35]
    }

    classifiers = {name: load_classifier(name) for name in [
        "RES50", "VGG19", "MOB_V2", "INC_V3", "CONVNEXT",
        "VIT_B", "SWIN_B", "DEIT_B", "DEIT_S", "MIX_B", "MIX_L",
    ]}

    evaluator = HyperparameterEvaluator(
        attack_class=MixedSignalAttack,#MixedResidualSignalAttack,
        pipe=pipe,
        classifiers=classifiers,
        evaluate_image_fn=evaluate,
        write_path="mixed_signal_attack_results/",
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