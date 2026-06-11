import logging
import torch
from scipy.stats import qmc
import pandas as pd
import numpy as np

from config import SEED, DEVICE, SAVE_DIR, set_seed
from models.classifiers import load_classifier
from evaluation.evaluate import evaluate, HyperparameterEvaluator
from attack.normal_gen import MixedSignalAttack, MixedResidualSignalAttack, MixedNormalResidualSignalAttack, MixedTimedResidualSignalAttack, BaseUNetGen, AttentionAttack

from diffusers import DDIMScheduler, DiffusionPipeline

s_values = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8])
t_values = np.array([15,20,25,30,35])
guidance_scales = np.array([5, 7.5, 10, 12.5, 15])

n_samples = 12

sampler = qmc.LatinHypercube(d=2)
lhs = sampler.random(n_samples)

s_idx = np.floor(lhs[:,0] * len(s_values)).astype(int)
t_idx = np.floor(lhs[:,1] * len(t_values)).astype(int)

pairs = set()
for si, ti in zip(s_idx, t_idx):
    pairs.add((s_values[si], t_values[ti]))

pairs = list(pairs)

all_pairs = [
    (s,t)
    for s in s_values
    for t in t_values
]

while len(pairs) < 12:
    candidate = all_pairs[np.random.randint(len(all_pairs))]
    pairs.append(candidate)
    pairs = list(set(pairs))
    
guidance_scales_arr = []
s_arr = []
adversarial_t_arr = []
for guidance in guidance_scales:
    for s, adv_t in pairs:
        guidance_scales_arr.append(guidance)
        s_arr.append(s)
        adversarial_t_arr.append(adv_t)

hyperparameter_grid={
    "guidance_scale": guidance_scales_arr,
    "s": s_arr,
    "adversarial_t": adversarial_t_arr,
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
    attack_class=MixedTimedResidualSignalAttack,#MixedResidualSignalAttack,
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
    's': s_arr,
    'adversarial_t': adversarial_t_arr,
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