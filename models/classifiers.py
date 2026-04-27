from torchvision import models

from config import DEVICE, INC_V3_WEIGHTS, VIT_B_32_WEIGHTS, SWIN_B_WEIGHTS

def load_classifier(model):
    if model == "INC_V3":
        return models.inception_v3(weights=INC_V3_WEIGHTS).to(DEVICE).eval()
    if model == "VIT_B":
        return models.vit_b_32(weights=VIT_B_32_WEIGHTS).to(DEVICE).eval()
    if model == "SWIN_B":
        return models.swin_b(weights=SWIN_B_WEIGHTS).to(DEVICE).eval()
    
    return ValueError("Model unavailable")