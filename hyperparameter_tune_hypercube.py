import logging
import torch
from scipy.stats import qmc
import pandas as pd
import numpy as np

from config import SEED, DEVICE, SAVE_DIR, set_seed
from models.classifiers import load_classifier
from evaluation.evaluate import evaluate, HyperparameterEvaluator
from attack.normal_gen import MixedSignalAttack, MixedResidualSignalAttack, MixedNormalResidualSignalAttack, MixedTimedResidualSignalAttack, NormalisedEnhancedMixedSignal, BaseUNetGen, AttentionAttack

from diffusers import DDIMScheduler, DiffusionPipeline

alpha_values = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8]) # 7 
beta_values = np.array([15,20,25,30,35]) # 5
guidance_scales = np.array([5, 7.5, 10, 12.5, 15])

n_samples = 12

sampler = qmc.LatinHypercube(d=2)
lhs = sampler.random(n_samples)

alpha_idx = np.floor(lhs[:,0] * len(alpha_values)).astype(int)
beta_idx = np.floor(lhs[:,1] * len(beta_values)).astype(int)

pairs = set()
for a_idx, b_idx in zip(alpha_idx, beta_idx):
    pairs.add((alpha_values[a_idx], beta_values[b_idx]))

pairs = list(pairs)

all_pairs = [
    (a,b)
    for a in alpha_values
    for b in beta_values
]

while len(pairs) < 12:
    candidate = all_pairs[np.random.randint(len(all_pairs))]
    pairs.append(candidate)
    pairs = list(set(pairs))
    
guidance_scales_arr = []
alpha_arr = []
beta_arr = []
for guidance in guidance_scales:
    for alpha, beta in pairs:
        guidance_scales_arr.append(guidance)
        alpha_arr.append(alpha)
        beta_arr.append(beta)

hyperparameter_grid={
    "guidance_scale": guidance_scales_arr,
    "alpha": alpha_arr,
    "beta": beta_arr,
    "adversarial_t": [0],
}

# 7 classes
dev_classes = [
    "castle",
    "acoustic guitar",
    "tabby",
    "king penguin",
    "basketball",
    "rocking chair",
    "tractor"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hyperparameter_tuning.log"),
    ]
)
log = logging.getLogger(__name__)

pipe = DiffusionPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-1-base",
    torch_dtype=torch.float32,
).to(DEVICE)

classifiers = {name: load_classifier(name) for name in [
    "RES50", "VGG19", "MOB_V2", "INC_V3", "CONVNEXT",
    "VIT_B", "SWIN_B", "DEIT_B", "DEIT_S", "MIX_B", "MIX_L",
]}

evaluator = HyperparameterEvaluator(
    attack_class=NormalisedEnhancedMixedSignal,#MixedResidualSignalAttack,
    pipe=pipe,
    classifiers=classifiers,
    evaluate_image_fn=evaluate,
    write_path="hp_tuning_results/",
    precomputed_latents=None,
    precomputed_labels=None,
    precomputed_fake_classes=None,
    precomputed_guidance_scales=None,
)

evaluator.run(
    classes=dev_classes,
    attacks_per_class=3,
    hyperparameter_grid=hyperparameter_grid,
    output_csv="results.csv",
)

df = pd.read_csv("hp_tuning_results/results.csv")
summary = df.groupby('guidance_scale').agg(
    avg_asr_targeted=('asr_targeted_global', 'mean'),
    avg_asr_untargeted=('asr_untargeted_global', 'mean'),
    avg_lpips=('lpips', 'mean'),
    avg_duration_s=('duration_s', 'mean'),
    n=('lpips', 'count')
).reset_index()
best = summary.sort_values(
    ['avg_asr_targeted', 'avg_asr_untargeted', 'avg_lpips'],
    ascending=[False, False, True]
).iloc[0]

best_guidance = best['guidance_scale']
new_hyperparameter_grid = {
    'guidance_score': [best_guidance],
    'alpha': alpha_arr,
    'beta': beta_arr,
    'adversarial_t': [0],
}

# 12 classes
classes = [
    "ambulance",
    "starfish",
    "sports car",
    "airliner",
    "toaster",
    "laptop",
    "stove",
    "beacon",
    "bald eagle",
    "Polaroid camera",
    "analog clock",
    "jellyfish",
    "microwave",
    "peacock",
    "vacuum",
    "tractor",
    "tabby"
]

evaluator = HyperparameterEvaluator(
    attack_class=MixedTimedResidualSignalAttack,#MixedResidualSignalAttack,
    pipe=pipe,
    classifiers=classifiers,
    evaluate_image_fn=evaluate,
    write_path="final_hp_tuning_results/",
    precomputed_latents=None,
    precomputed_labels=None,
    precomputed_fake_classes=None,
    precomputed_guidance_scales=None,
)

evaluator.run(
    classes=classes,
    attacks_per_class=3,
    hyperparameter_grid=new_hyperparameter_grid,
    output_csv="results.csv",
)