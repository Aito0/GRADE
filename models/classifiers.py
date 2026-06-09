from config import DEVICE

from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    VGG19_Weights,
    MobileNet_V2_Weights,
    Inception_V3_Weights,
    ConvNeXt_Base_Weights,
    ViT_B_32_Weights,
    Swin_B_Weights,
)
import timm

def load_classifier(model_name: str):
    match model_name:
        case "RES50":
            return models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        case "VGG19":
            return models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        case "MOB_V2":
            return models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        case "INC_V3":
            return models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        case "CONVNEXT":
            return models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        case "VIT_B":
            return models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        case "SWIN_B":
            return models.swin_b(weights=Swin_B_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        case "DEIT_B":
            return timm.create_model("deit_base_patch16_224",  pretrained=True).to(DEVICE).eval()
        case "DEIT_S":
            return timm.create_model("deit_small_patch16_224", pretrained=True).to(DEVICE).eval()
        case "MIX_B":
            return timm.create_model("mixer_b16_224",          pretrained=True).to(DEVICE).eval()
        case "MIX_L":
            return timm.create_model("mixer_l16_224",          pretrained=True).to(DEVICE).eval()
        case _:
            raise ValueError(f"Model '{model_name}' unavailable")