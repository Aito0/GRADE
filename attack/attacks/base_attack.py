import torch
import numpy as np

from abc import ABC, abstractmethod

from config import DEVICE
from utils import numpy_to_tensor
from data.imagenet import idx_to_label
from models.classifiers import load_classifier
from attack.attacks.normal_gen import NormalGen

class _BaseAttack(NormalGen, ABC):
    def __init__(
        self,
        tokeniser,
        vae,
        scheduler,
        unet,
        text_encoder,
        classifier=None,
    ):
        super().__init__(tokeniser, vae, scheduler, unet, text_encoder)
        self.classifier = (
            classifier.to(DEVICE)
            if classifier is not None
            else load_classifier("SWIN_B")
        )

    @abstractmethod
    def attack(self):
        pass

    @torch.no_grad()
    def get_top2_classes(self, image: np.ndarray) -> tuple[str, str]:
        logits = self.classifier(numpy_to_tensor(image).unsqueeze(0).to(DEVICE))
        top2 = torch.softmax(logits, dim=-1).topk(2).indices.squeeze()
        return idx_to_label(top2[0].item()), idx_to_label(top2[1].item())

    def generate_verified(
        self,
        true_class: str,
        time_steps: int,
        guidance_scale: float,
        latents: torch.Tensor | None = None,
        fake_class: str | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, str]:

        if fake_class is not None:
            latents = latents if latents is not None else self._fresh_latents()
            return latents, self.generate(true_class,time_steps,guidance_scale,latents), fake_class

        top_class = ""
        for i in range(10):
            latents = self._fresh_latents()
            clean_image = self.generate(true_class,time_steps,guidance_scale,latents)
            top_class, fake_class = self.get_top2_classes(clean_image)

            if top_class == true_class:
                break

        if top_class != true_class:
            return None, None, None
        
        return latents, clean_image, fake_class