import os
import numpy as np
from PIL import Image
from config import DATASET_PATH, INC_V3_WEIGHTS

def _load_imagenet_mini_labels(weights):
    idx_to_label, folder_to_label, label_to_idx = {}, {}, {}
    val_folders = sorted(os.listdir(DATASET_PATH))
    categories = weights.meta['categories']
    for (idx, category), val_folder in zip(enumerate(categories), val_folders):
        idx_to_label[idx] = category
        folder_to_label[val_folder] = category
        label_to_idx[category] = idx
    return idx_to_label, folder_to_label, label_to_idx

_idx_to_label, _folder_to_label, _label_to_idx = _load_imagenet_mini_labels(INC_V3_WEIGHTS)

def index_to_label(idx):
    return _idx_to_label[idx]

def folder_to_label(idx):
    return _folder_to_label[idx]

def label_to_idx(label):
    return _label_to_idx[label]

def select_images(folder_to_label):
    val_folders = os.listdir(DATASET_PATH)
    folders = np.random.choice(val_folders, size=2, replace=False)

    def _pick_image_from_folder(folder):
        folder_path = os.path.join(DATASET_PATH, folder)
        image_file = np.random.choice(os.listdir(folder_path))
        img = Image.open(os.path.join(folder_path, image_file))
        img = img.resize((256, 256)).crop((16, 16, 240, 240))
        img = np.array(img) / 255
        label = folder_to_label[folder]
        return img, label

    return [_pick_image_from_folder(f) for f in folders]