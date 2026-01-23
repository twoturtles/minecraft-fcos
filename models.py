from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import torchvision as tv  # type: ignore
from IPython.display import display
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import fcos  # type: ignore
from torchvision.transforms import v2 as v2  # type: ignore
from tqdm.auto import tqdm, trange

import bb


class FCOSTrainer:

    def __init__(
        self,
        *,
        categories: list[str],
        checkpoint: Path | str | None = None,
        device: str | torch.device = "mps",
        he_init: bool = True,
    ) -> None:
        self.categories = categories
        self.device = torch.device(device)

        self.preprocess = fcos.FCOS_ResNet50_FPN_Weights.COCO_V1.transforms()

        if checkpoint is None:
            self._load_pretrained(he_init=he_init)
        else:
            self._load_checkpoint(checkpoint)

    def _load_pretrained(self, he_init: bool = True) -> None:
        """Load FCOS pretrained weights.
        This is expected: missing_keys=[
            "head.classification_head.cls_logits.weight",
            "head.classification_head.cls_logits.bias",
        ],
        he_init - Use He Kaiming init rather than standard FCOS init
        """
        print("Initializing new model")
        # FCOS init
        self.model = fcos.fcos_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=len(self.categories)
        )
        # Load pretrained
        model_state_dict = fcos.FCOS_ResNet50_FPN_Weights.COCO_V1.get_state_dict(
            progress=True, check_hash=True
        )

        # Set up classification head init
        if he_init:
            num_classes = len(self.categories)
            # Original: Conv2d(256, 91, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            cls_logits = torch.nn.Conv2d(
                256, num_classes, kernel_size=3, stride=1, padding=1
            )
            state = cls_logits.state_dict()
            model_state_dict["head.classification_head.cls_logits.weight"] = state[
                "weight"
            ]
            model_state_dict["head.classification_head.cls_logits.bias"] = state["bias"]
        else:
            # Don't try to apply 91 class init to our model head. Leave FCOS init.
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

    def filter_pred(
        self, pred: dict[str, Any], score_thresh: float = 0.5
    ) -> dict[str, Any]:
        mask = pred["scores"] >= score_thresh
        filtered: dict[str, Any] = {}
        for key, vals in pred.items():
            filtered[key] = vals[mask]

        return filtered

    def filter_preds(
        self, preds: list[dict[str, Any]], score_thresh: float = 0.5
    ) -> list[dict[str, Any]]:
        return [self.filter_pred(pred, score_thresh) for pred in preds]

    def infer(
        self, img: tv.tv_tensors.Image, score_thresh: float = 0.5
    ) -> dict[str, Any]:
        self.model.eval()
        img = img.to(self.device)
        batch = [self.preprocess(img)]
        with torch.inference_mode():
            pred: dict[str, Any] = self.model(batch)[0]
        pred = self.filter_pred(pred, score_thresh)
        return pred

    def forward(
        self, batch: torch.Tensor, score_thresh: float = 0.5
    ) -> list[dict[str, Any]]:
        self.model.eval()
        batch = self.preprocess(batch.to(self.device))
        with torch.inference_mode():
            preds: list[dict[str, Any]] = self.model(batch)
        preds = self.filter_preds(preds, score_thresh)
        return preds

    def plot_infer(
        self, img: tv.tv_tensors.Image, score_thresh: float = 0.5
    ) -> Image.Image:
        pred = self.infer(img, score_thresh)
        ret = bb.torch_plot_bb(
            img, pred, self.categories, include_scores=True, return_pil=True
        )
        assert isinstance(ret, Image.Image)
        return ret

    def _plot_loss(self, figsize: tuple[int, int] = (12, 3)) -> plt.Figure:
        """Create loss figure. Returns figure for caller to display/handle."""
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

        train_x = np.linspace(0, self.total_epochs, len(self.loss_log))
        ax.plot(train_x, self.loss_log)
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        return fig

    def plot_loss(self, figsize: tuple[int, int] = (12, 3)) -> None:
        """One-off plot display."""
        fig = self._plot_loss(figsize)
        plt.show()
        plt.close(fig)

    def train(self, train_loader: DataLoader[Any], num_epochs: int) -> None:
        """Train with live-updating plot."""
        fig = self._plot_loss()
        handle = display(fig, display_id=True)  # type: ignore
        plt.close(fig)

        for epoch in trange(num_epochs, leave=True, desc="Epoch"):
            self.train_one_epoch(train_loader)
            fig = self._plot_loss()
            handle.update(fig)
            plt.close(fig)

    def _fixup_targets(self, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Handle images with no boxes and move to device."""
        fixed = [
            {
                "boxes": t.get("boxes", torch.zeros(0, 4)).to(self.device),
                "labels": t.get("labels", torch.zeros(0, dtype=torch.int64)).to(
                    self.device
                ),
            }
            for t in targets
        ]
        return fixed

    def train_one_epoch(self, train_loader: DataLoader[Any]) -> None:
        self.model.train()

        for images, targets in tqdm(train_loader, leave=False, desc="Batch"):
            images = images.to(self.device)
            targets = self._fixup_targets(targets)

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

    def evaluate(self, val_loader: DataLoader[Any]) -> torchmetrics.Metric:
        self.model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")

        with torch.inference_mode():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = self._fixup_targets(targets)
                batch = self.preprocess(images)
                preds = self.model(batch)
                metric.update(preds, targets)

        return {"metric": metric, "images": images, "preds": preds, "targets": targets}


def compare_models(
    m0: torch.nn.Module, m1: torch.nn.Module, exact: bool = True
) -> bool:
    """Check if two models have identical parameters."""
    sd0 = m0.state_dict()
    sd1 = m1.state_dict()

    if sd0.keys() != sd1.keys():
        return False

    check_fn = torch.equal if exact else torch.allclose
    for key in sd0:
        if not check_fn(sd0[key], sd1[key]):
            print(f"Mismatch at: {key}")
            return False
    return True
