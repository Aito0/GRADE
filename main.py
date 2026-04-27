from config import SEED, DEVICE, SAVE_DIR, set_seed
from models.classifiers import load_classifier
from models.diffusion import load_pipeline, load_scheduler
from attack.advdiff import AdvDiffAttack
from evaluation.evaluate import evaluate

import numpy as np
import os

def main():
    set_seed(SEED)
    
    classifier = load_classifier("SWIN_B")
    
    attack = AdvDiffAttack() # MixAttack.from_pipe(pipe).generate_examples(classes=5, gen_per_class=3)
    imgs, labels = attack.generate_examples()
    
    save_img = imgs.permute(0,2,3,1)

    probs = []
    pred_classes = []
    for img in imgs:
        pred_class, prob = evaluate(img, classifier)
        pred_classes.append(pred_class)
        probs.append(prob)
    
    np.savez(os.path.join(SAVE_DIR, 'mix.npz'), save_img.detach().cpu().numpy(), labels.detach().cpu().numpy(), pred_classes, probs)

if __name__ == "__main__":
    main()