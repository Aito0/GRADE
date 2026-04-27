import torch
from torchvision.models import Inception_V3_Weights, ViT_B_32_Weights, Swin_B_Weights

import os
import random
import numpy as np
from dotenv import load_dotenv

load_dotenv()

torch.cuda.empty_cache()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = int(os.getenv('SEED'))
STABLE_DIFFUSION_DIR = os.getenv('STABLE_DIFFUSION_DIR')
DATASET_PATH = os.getenv('IMAGENET_MINI_DIR')
SAVE_DIR = os.getenv("SAVE_DIR")

# ImageNet weights
INC_V3_WEIGHTS = Inception_V3_Weights.DEFAULT
VIT_B_32_WEIGHTS = ViT_B_32_Weights.DEFAULT
SWIN_B_WEIGHTS = Swin_B_Weights.DEFAULT

TIME_STEPS = 50
GUIDANCE_SCALE = 7.5
STRENGTH = 1.0
N = 10
A = 0.5

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)