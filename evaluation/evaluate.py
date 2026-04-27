import torch
from utils import numpy_to_tensor
from ..data.imagenet import index_to_label
from config import DEVICE

def evaluate(image, classifier):
    pred_image = numpy_to_tensor(image).unsqueeze(0).to(DEVICE)
    logits = torch.nn.functional.softmax(classifier(pred_image)[0], dim=0)
    pred_class = torch.argmax(logits).item()
    return index_to_label(pred_class), logits[pred_class].item()