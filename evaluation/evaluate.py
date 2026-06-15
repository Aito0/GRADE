import os
import time
import lpips
import torch
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import logging

from utils import numpy_to_tensor
from data.imagenet import idx_to_label, label_to_idx
from config import DEVICE

import torch
import torchvision.transforms as T

from models.classifiers import load_classifier

log = logging.getLogger(__name__)

class LPIPS:
    model = None

    @classmethod
    def get_lpips_model(cls):
        if cls.model is None:
            cls.model = lpips.LPIPS(net='alex').to(DEVICE)
            cls.model.eval()
        return cls.model


def evaluate(image, true_class=None, fake_class=None, model=None):
    if isinstance(image, np.ndarray):
        image = numpy_to_tensor(image).unsqueeze(0)
        
    preds       = model(image.to(next(model.parameters()).device))
    pred_logits = torch.nn.functional.softmax(preds[0], dim=0)

    true_class_prob = pred_logits[label_to_idx(true_class)].item() if true_class else 0.0
    fake_class_prob = pred_logits[label_to_idx(fake_class)].item() if fake_class else 0.0

    pred_idx   = torch.argmax(pred_logits).item()
    pred_class = idx_to_label(pred_idx)
    pred_prob  = pred_logits[pred_idx].item()

    return pred_class, pred_prob, true_class_prob, fake_class_prob


def lpips_score(adv_image, clean_image):
    t1 = numpy_to_tensor(adv_image).unsqueeze(0).to(DEVICE)
    t2 = numpy_to_tensor(clean_image).unsqueeze(0).to(DEVICE)

    t1 = t1 * 2 - 1
    t2 = t2 * 2 - 1

    with torch.no_grad():
        score = LPIPS.get_lpips_model()(t1, t2)  # get model first, then call it

    return score.item()

MODEL_SIZES = {
    "RES50": 224,
    "VGG19": 224,
    "MOB_V2": 224,
    "CONVNEXT": 224,

    "VIT_B": 224,
    "SWIN_B": 224,
    "DEIT_B": 224,
    "DEIT_S": 224,
    "MIX_B": 224,
    "MIX_L": 224,

    "INC_V3": 299,   # only real exception
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class Results:
    def __init__(self):
        self.pred_classes     = []
        self.pred_probs       = []
        self.true_class_probs = []
        self.fake_class_probs = []

    def add(self, pred_class, pred_prob, true_class_prob, fake_class_prob):
        self.pred_classes.append(pred_class)
        self.pred_probs.append(pred_prob)
        self.true_class_probs.append(true_class_prob)
        self.fake_class_probs.append(fake_class_prob)

# Class originally written by me, but extended later to allow precomputed information using GenAI
class HyperparameterEvaluator:

    def __init__(self, attack_class, pipe, classifiers, evaluate_image_fn,
                 write_path="results/",
                 precomputed_latents=None,
                 precomputed_labels=None,
                 precomputed_fake_classes=None,
                 precomputed_guidance_scales=None):
        self.attack_class      = attack_class
        self.pipe              = pipe
        self.classifiers       = classifiers
        self.evaluate_image_fn = evaluate_image_fn
        self.write_path        = write_path
        os.makedirs(write_path, exist_ok=True)

        self._latent_pool     = {}
        self._latent_counters = {}

        if precomputed_latents is not None:
            for latent, label, fake_class, gs in zip(
                precomputed_latents, precomputed_labels,
                precomputed_fake_classes, precomputed_guidance_scales,
            ):
                if latent is None:
                    continue
                key = (label, float(gs))
                self._latent_pool.setdefault(key, []).append((latent, fake_class))

            self._latent_counters = {k: 0 for k in self._latent_pool}

            total_latents = sum(len(pool) for pool in self._latent_pool.values())
            log.info(f"Loaded {total_latents} latents across "
                     f"{len(self._latent_pool)} (label, guidance_scale) pairs")

        log.info(f"Classifiers: {list(self.classifiers.keys())}")
        log.info(f"Attack class: {self.attack_class.__name__}")
        log.info(f"Write path: {self.write_path}")

    def _save_path(self, true_class, hyperparams, index, clean=False) -> str:
        top_level_dir = "images" if not clean else "clean_images"
        hyperparam_str = "_".join([f"{k}{v}" for k, v in hyperparams.items()])
        folder = os.path.join(self.write_path, top_level_dir, true_class.replace(" ", "_"), hyperparam_str)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{index}.png")

    def _save_image(self, img: np.ndarray, path: str):
        img_norm = (img - img.min()) / (img.max() - img.min()) * 255
        Image.fromarray(img_norm.astype(np.uint8)).save(path)

    def _append_csv(self, row: dict, csv_path: str):
        pd.DataFrame([row]).to_csv(
            csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    def _evaluate_image(self, img, true_class, fake_class) -> dict:
        results = {}
        pil_img = Image.fromarray(img)
        
        for model_name, model in self.classifiers.items():
            img = T.Compose([
                T.Resize(256),
                T.CenterCrop(MODEL_SIZES[model_name]),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])(pil_img).unsqueeze(0)
            try:
                pred_class, pred_prob, true_class_prob, fake_class_prob = self.evaluate_image_fn(
                    img, true_class, fake_class, model=model,
                )
            
                results[model_name] = {
                    "pred_class":      pred_class,
                    "pred_prob":       pred_prob,
                    "true_class_prob": true_class_prob,
                    "fake_class_prob": fake_class_prob,
                    "asr_targeted":    int(pred_class == fake_class),
                    "asr_untargeted":  int(pred_class != true_class),
                }
            except Exception as e:
                print(f"Model: {model_name} can't take that image size")
                raise e
        return results

    def _build_row(self, store_path, true_class, fake_class,
                   model_results, hyperparams, lpips_val, duration) -> dict:
        row = {
            "store_path": store_path,
            "true_class": true_class,
            "fake_class": fake_class,
            "lpips":      lpips_val,
            "duration_s": duration,
            "skipped":    False,
            **hyperparams,
        }
        targeted_asrs   = []
        untargeted_asrs = []
        for model_name, res in model_results.items():
            for metric, val in res.items():
                row[f"{model_name}_{metric}"] = val
            targeted_asrs.append(res["asr_targeted"])
            untargeted_asrs.append(res["asr_untargeted"])

        row["asr_targeted_global"]   = int(all(targeted_asrs))
        row["asr_untargeted_global"] = int(all(untargeted_asrs))
        return row

    def _build_skipped_row(self, store_path, true_class, fake_class, hyperparams) -> dict:
        row = {
            "store_path": store_path,
            "true_class": true_class,
            "fake_class": fake_class,
            "lpips":      None,
            "duration_s": None,
            "skipped":    True,
            **hyperparams,
        }
        for model_name in self.classifiers:
            row[f"{model_name}_pred_class"]      = None
            row[f"{model_name}_pred_prob"]       = None
            row[f"{model_name}_true_class_prob"] = None
            row[f"{model_name}_fake_class_prob"] = None
            row[f"{model_name}_asr_targeted"]    = None
            row[f"{model_name}_asr_untargeted"]  = None
        row["asr_targeted_global"]   = None
        row["asr_untargeted_global"] = None
        return row

    def _next_latent(self, true_class, guidance_scale):
        key  = (true_class, float(guidance_scale))
        pool = self._latent_pool.get(key, [])
        if not pool:
            return None, None
        idx                = self._latent_counters[key]
        latent, fake_class = pool[idx % len(pool)]
        self._latent_counters[key] += 1
        return latent, fake_class

    def run(self, classes, hyperparameter_grid, attacks_per_class, output_csv="results.csv"):
        keys     = list(hyperparameter_grid.keys())
        values   = list(hyperparameter_grid.values())
        combos   = list(itertools.product(*values))
        csv_path = os.path.join(self.write_path, output_csv)

        total              = len(classes) * len(combos) * attacks_per_class
        primary_classifier = load_classifier("SWIN_B") # next(iter(self.classifiers.values()))

        log.info(f"Starting evaluation — {total} total attacks")
        log.info(f"Classes: {classes}")
        log.info(f"Hyperparameter grid: {hyperparameter_grid}")
        log.info(f"Attacks per class: {attacks_per_class}")

        completed = skipped = errors = already_exists = 0

        with tqdm(total=total, desc="Evaluations") as pbar:
            for true_class in classes:
                for combo in combos:
                    hyperparams    = dict(zip(keys, combo))
                    guidance_scale = float(hyperparams["guidance_scale"])
                    self._latent_counters[(true_class, guidance_scale)] = 0

                    for i in range(attacks_per_class):
                        latent, fake_class = self._next_latent(true_class, guidance_scale)
                        store_path         = self._save_path(true_class, hyperparams, i)
                        clean_store_path   = self._save_path(true_class, hyperparams, i, clean=True)

                        if os.path.isfile(store_path):
                            already_exists += 1
                            pbar.update(1)
                            continue


                        try:
                            attack = self.attack_class.load_from_pipe(
                                self.pipe, classifier=primary_classifier)

                            t0 = time.time()
                            attacked_img, fake_class, clean_img = attack.run_attack(
                                true_class,
                                latents=latent,
                                fake_class=fake_class,
                                **hyperparams,
                            )
                            duration = time.time() - t0
                            
                            if clean_img is None:
                                log.warning(f"No latent for label={true_class}, "
                                            f"guidance_scale={guidance_scale}, i={i} — writing null row")
                                row = self._build_skipped_row(store_path, true_class, fake_class, hyperparams)
                                self._append_csv(row, csv_path)
                                skipped += 1
                                pbar.update(1)
                                continue

                            model_results = self._evaluate_image(attacked_img, true_class, fake_class)
                            lpips_val     = lpips_score(attacked_img, clean_img)

                            asr_t = {m: res["asr_targeted"]   for m, res in model_results.items()}
                            asr_u = {m: res["asr_untargeted"] for m, res in model_results.items()}
                            log.info(f"[{completed+skipped+errors+1}/{total}] "
                                     f"class={true_class}, fake={fake_class}, params={hyperparams} | "
                                     f"asr_t={asr_t}, asr_u={asr_u}, "
                                     f"lpips={lpips_val:.4f}, duration={duration:.1f}s")

                            self._save_image(attacked_img, store_path)
                            self._save_image(clean_img, clean_store_path)
                            row = self._build_row(
                                store_path, true_class, fake_class,
                                model_results, hyperparams, lpips_val, duration,
                            )
                            self._append_csv(row, csv_path)
                            completed += 1

                        except Exception as e:
                            log.error(f"Error — class={true_class}, params={hyperparams}, "
                                      f"i={i}: {e}", exc_info=True)
                            errors += 1

                        pbar.update(1)

        log.info(f"Evaluation complete — completed={completed}, skipped={skipped}, "
                 f"errors={errors}, already_existed={already_exists}")