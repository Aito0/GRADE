import os
import torch
import optuna
import numpy as np
import pandas as pd

from config import SEED, DEVICE, SAVE_DIR, set_seed
from models.classifiers import load_classifier
from evaluation.evaluate import evaluate, HyperparameterEvaluator
from attack.normal_gen import MixedSignalAttack, MixedResidualSignalAttack, MixedNormalResidualSignalAttack, MixedTimedResidualSignalAttack, BaseUNetGen, AttentionAttack

from diffusers import DDIMScheduler, DiffusionPipeline

SCREENING_CLASSES = [
    "golden retriever",
    "tabby cat",
    "sports car",
    "fire engine",
    "airliner",
    "banana",
    "church",
    "castle",
    "volcano",
    "espresso"
]

REPEATS = 2

def make_objective(evaluator, classes, repeats=2):
    def objective(trial):
        hyperparams = {
            "guidance_scale": trial.suggest_categorical(
                "guidance_scale",
                [7.5, 10, 12.5, 15]
            ),

            "alpha": trial.suggest_float(
                "alpha",
                0.2,
                0.8
            ),

            "adversarial_t": trial.suggest_int(
                "adversarial_t",
                15,
                35,
                step=5
            ),

            "blocks_changed": trial.suggest_int(
                "blocks_changed",
                1,
                3,
                step=1
            ),
        }

        run_id = f"trial_{trial.number}"
        csv_path = f"{run_id}.csv"

        evaluator.run(
            classes=classes,
            hyperparameter_grid={k: [v] for k, v in hyperparams.items()},
            output_csv=csv_path,
            attacks_per_class=1,
        )

        df = pd.read_csv(os.path.join(evaluator.write_path, csv_path))

        # IMPORTANT: robust aggregation
        avg_asr = df["asr_targeted_global"].mean()
        avg_lpips = df["lpips"].mean()

        return avg_asr, avg_lpips

    return objective


pipe = DiffusionPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-1-base",
    torch_dtype=torch.float32,
).to(DEVICE)

classifiers = {name: load_classifier(name) for name in [
    "RES50", "VGG19", "MOB_V2", "INC_V3", "CONVNEXT",
    "VIT_B", "SWIN_B", "DEIT_B", "DEIT_S", "MIX_B", "MIX_L",
]}

# Also run with attack_class = AttentionAttack
evaluator = HyperparameterEvaluator(
    attack_class=BaseUNetGen,
    pipe=pipe,
    classifiers=classifiers,
    evaluate_image_fn=evaluate,
    write_path="test_results/",
    precomputed_latents=None,
    precomputed_labels=None,
    precomputed_fake_classes=None,
    precomputed_guidance_scales=None,
)

study = optuna.create_study(
    directions=["maximize", "minimize"]
)

study.optimize(
    make_objective(evaluator, classes=SCREENING_CLASSES),
    n_trials=50
)

df = study.trials_dataframe()

df.to_csv(
    "optuna_skip_attack.csv",
    index=False
)