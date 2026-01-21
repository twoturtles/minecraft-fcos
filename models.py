from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv  # type: ignore
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection import fcos  # type: ignore
from torchvision.transforms import v2 as v2  # type: ignore
from tqdm.auto import tqdm, trange


class FCOSTrainer:

    def __init__(
        self,
        *,
        categories: list[str],
        checkpoint: Path | str | None = None,
        device: str | torch.device = "mps",
    ) -> None:
        self.categories = categories
        self.device = torch.device(device)

        self.preprocess = fcos.FCOS_ResNet50_FPN_Weights.COCO_V1.transforms()

        if checkpoint is None:
            self._load_pretrained()
        else:
            self._load_checkpoint(checkpoint)

    def _load_pretrained(self) -> None:
        """Load FCOS pretrained weights."""
        print("Initializing new model")
        self.model = fcos.fcos_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=len(self.categories)
        )
        model_state_dict = fcos.FCOS_ResNet50_FPN_Weights.COCO_V1.get_state_dict(
            progress=True, check_hash=True
        )
        del model_state_dict["head.classification_head.cls_logits.weight"]
        del model_state_dict["head.classification_head.cls_logits.bias"]
        self._setup_model(model_state_dict=model_state_dict)

    def _load_checkpoint(self, ckpt_file: Path | str) -> None:
        ckpt_file = Path(ckpt_file)
        print(f"Loading checkpoint: {ckpt_file}")
        self.model = fcos.fcos_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=len(self.categories)
        )
        ckpt = torch.load(ckpt_file, weights_only=True)
        self._setup_model(
            model_state_dict=ckpt["model_state_dict"],
            optimizer_state_dict=ckpt["optimizer_state_dict"],
            total_epochs=ckpt["total_epochs"],
            loss_log=ckpt["loss_log"],
        )

    def _setup_model(
        self,
        *,
        model_state_dict: dict[str, Any],
        optimizer_state_dict: dict[str, Any] | None = None,
        total_epochs: int = 0,
        loss_log: list[float] | None = None,
    ) -> None:
        self.total_epochs = total_epochs
        self.loss_log = loss_log if loss_log else []

        err_keys = self.model.load_state_dict(model_state_dict, strict=False)
        print(f"err_keys = {err_keys}")
        self.model.to(self.device)
        self._set_requires_grad()

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=1e-4)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

    def save_checkpoint(self, ckpt_file: Path | str) -> None:
        ckpt_file = Path(ckpt_file)
        checkpoint = {
            "total_epochs": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # "scheduler_state_dict": scheduler.state_dict(),
            "loss_log": self.loss_log,
        }
        torch.save(checkpoint, ckpt_file)

    def _set_requires_grad(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.model.head.classification_head.cls_logits.parameters():
            param.requires_grad_(True)

    def infer(self, img: tv.tv_tensors.Image) -> Image.Image:
        self.model.eval()
        img = img.to(self.device)
        batch = [self.preprocess(img)]
        with torch.inference_mode():
            prediction = self.model(batch)[0]
        labels = [self.categories[i] for i in prediction["labels"]]
        box = tv.utils.draw_bounding_boxes(
            img,
            boxes=prediction["boxes"],
            labels=labels,
            colors="red",
            width=4,
            font="/System/Library/Fonts/Helvetica.ttc",  # macOS
            font_size=20,
        )
        ret: Image.Image = v2.functional.to_pil_image(box.detach())
        return ret

    def plot_loss(self, figsize: tuple[int, int] = (8, 6)) -> None:
        plt.figure(figsize=figsize)
        train_x = np.linspace(0, self.total_epochs, len(self.loss_log))
        plt.plot(train_x, self.loss_log)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def train(self, train_loader: DataLoader[Any], num_epochs: int) -> None:
        for epoch in trange(num_epochs, leave=True, desc="Epoch"):
            self.train_one_epoch(train_loader)
            self.plot_loss(figsize=(12, 3))

    def train_one_epoch(self, train_loader: DataLoader[Any]) -> None:
        self.model.train()

        for images, targets in tqdm(train_loader, leave=False, desc="Batch"):
            images = images.to(self.device)
            targets = [
                {
                    # Handle images with no boxes
                    "boxes": t.get("boxes", torch.zeros(0, 4)).to(self.device),
                    "labels": t.get("labels", torch.zeros(0, dtype=torch.int64)).to(
                        self.device
                    ),
                }
                for t in targets
            ]

            # Forward pass of image through network and get output
            batch = self.preprocess(images)
            # torchvision models return loss in train mode.
            loss_dict = self.model(batch, targets)
            loss = sum(loss_dict.values())
            self.loss_log.append(loss.item())

            # Zero gradients
            self.optimizer.zero_grad()

            # Backpropagate gradients
            loss.backward()
            # Do a single optimization step
            self.optimizer.step()

        self.total_epochs += 1
