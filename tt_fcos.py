"""
Class to help with training FCOS
"""

from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv  # type: ignore
from IPython.display import display
from matplotlib.figure import Figure
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision import tv_tensors
from torchvision.models.detection import fcos  # type: ignore
from torchvision.transforms import v2 as v2  # type: ignore
from tqdm.auto import tqdm, trange

import bb


class Meta(TypedDict, total=False):
    classes: list[str]
    total_epochs: int
    best_map: float
    best_epoch: int
    loss_log: list[float]
    eval_log: list[dict[str, Any]]
    lr_log: list[float]


class FCOSTrainer:

    def __init__(
        self,
        *,
        meta: Meta,
        project_dir: Path | str,
        model_state_dict: dict[str, Any],
        optimizer_state_dict: dict[str, Any] | None = None,
        scheduler_state_dict: dict[str, Any] | None = None,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.4,
        device: str | torch.device = "mps",
    ) -> None:
        # Initialize meta with defaults
        self.meta: Meta = {
            "classes": meta["classes"],
            "total_epochs": meta.get("total_epochs", 0),
            "best_map": meta.get("best_map", 0.0),
            "best_epoch": meta.get("best_epoch", 0),
            "loss_log": meta.get("loss_log") or [],
            "eval_log": meta.get("eval_log") or [],
            "lr_log": meta.get("lr_log") or [],
        }

        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = self.project_dir.name
        self.device = torch.device(device)
        self.best_checkpoint = self.project_dir / "best.pt"
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        self.preprocess = fcos.FCOS_ResNet50_FPN_Weights.COCO_V1.transforms()

        # Create and setup model
        self.model = fcos.fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=len(self.meta["classes"]),
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
        )
        err_keys = self.model.load_state_dict(model_state_dict, strict=False)
        print(f"err_keys = {err_keys}")
        self.model.to(self.device)
        self._set_requires_grad()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(params=self.model.parameters())
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

        # Setup scheduler
        # XXX Manually set T_max for LR scheduler to num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        if scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)

    @classmethod
    def new(
        cls,
        *,
        classes: list[str],
        project_dir: Path | str,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.4,
        device: str | torch.device = "mps",
    ) -> "FCOSTrainer":
        """Create a new FCOSTrainer with pretrained COCO weights.

        Missing keys for classification head are expected since we use
        a different number of classes.
        """
        print("Initializing new model")
        model_state_dict = fcos.FCOS_ResNet50_FPN_Weights.COCO_V1.get_state_dict(
            progress=True, check_hash=True
        )
        # Remove classification head weights (different num_classes)
        del model_state_dict["head.classification_head.cls_logits.weight"]
        del model_state_dict["head.classification_head.cls_logits.bias"]

        return cls(
            meta={"classes": classes},
            project_dir=project_dir,
            model_state_dict=model_state_dict,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            device=device,
        )

    @classmethod
    def load_checkpoint(
        cls,
        ckpt_file: Path | str | int,
        *,
        project_dir: Path | str,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.4,
        device: str | torch.device = "mps",
    ) -> "FCOSTrainer":
        """Load an FCOSTrainer from a checkpoint file."""
        project_dir_path = Path(project_dir)

        if isinstance(ckpt_file, int):
            ckpt_file = project_dir_path / f"ep-{ckpt_file}.pt"
        else:
            ckpt_file = project_dir_path / ckpt_file

        print(f"Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, weights_only=True)

        return cls(
            meta=ckpt["meta"],
            project_dir=project_dir,
            model_state_dict=ckpt["model_state_dict"],
            optimizer_state_dict=ckpt["optimizer_state_dict"],
            scheduler_state_dict=ckpt["scheduler_state_dict"],
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            device=device,
        )

    def save_checkpoint(self, ckpt_file: Path | str | None = None) -> None:
        ckpt_file = (
            self.project_dir / f"ep-{self.meta['total_epochs']}.pt"
            if ckpt_file is None
            else Path(ckpt_file)
        )
        checkpoint = {
            "meta": self.meta,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, ckpt_file)

    def _set_requires_grad(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.model.head.parameters():
            param.requires_grad_(True)
        for param in self.model.backbone.fpn.parameters():
            param.requires_grad_(True)

    def infer(self, img: tv.tv_tensors.Image) -> bb.Detection:
        batch = img.unsqueeze(0)
        return self.forward(batch)[0]

    def forward(self, batch: torch.Tensor) -> list[bb.Detection]:
        self.model.eval()
        batch = self.preprocess(batch.to(self.device))
        h, w = batch.shape[2:]
        with torch.inference_mode():
            raw_preds: list[dict[str, Any]] = self.model(batch)
        # Convert to typed for clarity
        preds: list[bb.Detection] = []
        for raw_pred in raw_preds:
            pred = bb.Detection(
                boxes=tv_tensors.BoundingBoxes(
                    raw_pred["boxes"],
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(h, w),
                ),
                scores=raw_pred["scores"],
                labels=raw_pred["labels"],
            )
            preds.append(pred)
        return preds

    def train(
        self,
        *,
        num_epochs: int,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None = None,
    ) -> None:
        """Train with live-updating plot."""
        fig = self.plot_loss(show=False)
        loss_handle = display(fig, display_id=True)  # type: ignore
        plt.close(fig)
        fig = self.plot_eval(show=False)
        eval_handle = display(fig, display_id=True)  # type: ignore
        plt.close(fig)

        for epoch in trange(num_epochs, leave=True, desc="Epoch"):
            self.train_one_epoch(train_loader=train_loader, val_loader=val_loader)

            fig = self.plot_loss(show=False)
            loss_handle.update(fig)
            plt.close(fig)
            fig = self.plot_eval(show=False)
            eval_handle.update(fig)
            plt.close(fig)

        print(
            f"Final epochs={self.meta['total_epochs']} loss={self.meta['loss_log'][-1]:.4f} mAP={self.meta['eval_log'][-1]['map']:.4f}"
        )

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

    def train_one_epoch(
        self,
        *,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None = None,
    ) -> None:
        self.model.train()

        for images, targets in tqdm(train_loader, leave=False, desc="Batch"):
            images = images.to(self.device)
            targets = self._fixup_targets(targets)

            # Forward pass of image through network and get output
            batch = self.preprocess(images)
            # torchvision models return loss in train mode.
            loss_dict = self.model(batch, targets)
            loss = sum(loss_dict.values())
            self.meta["loss_log"].append(loss.item())

            # Zero gradients
            self.optimizer.zero_grad()

            # Backpropagate gradients
            loss.backward()
            # Do a single optimization step
            self.optimizer.step()

        self.meta["total_epochs"] += 1
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.meta["lr_log"].append(current_lr)
        self.scheduler.step()

        if val_loader is not None:
            metrics = self.evaluate(val_loader)
            self.meta["eval_log"].append(metrics)
            # self.scheduler.step(metrics["map"])  # For ReduceLROnPlateau
            if metrics["map"] > self.meta["best_map"]:
                self.meta["best_map"] = metrics["map"]
                self.meta["best_epoch"] = self.meta["total_epochs"]
                print(
                    f"New best mAP={self.meta['best_map']:.4f} at epoch {self.meta['best_epoch']}"
                )
                self.save_checkpoint(self.best_checkpoint)

        print(
            f"Epoch {self.meta['total_epochs']}: val mAP={metrics['map']:.4f} lr={current_lr:.6f}"
        )

    def evaluate(self, loader: DataLoader[Any]) -> dict[str, Any]:
        self.model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")

        with torch.inference_mode():
            for images, targets in tqdm(loader, leave=False, desc="Eval"):
                images = images.to(self.device)
                targets = self._fixup_targets(targets)
                batch = self.preprocess(images)
                preds = self.model(batch)
                metric.update(preds, targets)

        return metric.compute()

    ## Plot Utils

    def plot_infer(self, img: tv.tv_tensors.Image) -> Image.Image:
        pred = self.infer(img)
        ret = bb.torch_plot_bb(
            img, pred, self.meta["classes"], include_scores=True, return_pil=True
        )
        assert isinstance(ret, Image.Image)
        return ret

    def plot_loss(
        self,
        figsize: tuple[int, int] = (12, 3),
        label: str = "",
        epoch_range: tuple[int | None, int | None] | None = None,
        show: bool = False,
    ) -> Figure:
        """Create loss figure. Returns figure for caller to display/handle."""
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

        loss_log = self.meta["loss_log"]
        total_epochs = self.meta["total_epochs"]

        if epoch_range is None:
            iter_slice = slice(None)
        else:
            iters_per_epoch = len(loss_log) // total_epochs
            start = (
                epoch_range[0] * iters_per_epoch if epoch_range[0] is not None else None
            )
            end = (
                epoch_range[1] * iters_per_epoch if epoch_range[1] is not None else None
            )
            iter_slice = slice(start, end)

        train_x = np.linspace(0, total_epochs, len(loss_log))[iter_slice]
        loss_log = loss_log[iter_slice]

        ax.plot(train_x, loss_log)
        ax.set_title(f"Training Loss {label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.close(fig)
        if show:
            display(fig)  # type: ignore
        return fig

    def plot_eval(
        self,
        keys: list[str] | None = None,
        figsize: tuple[int, int] = (12, 3),
        label: str = "",
        epoch_range: tuple[int | None, int | None] | None = None,
        show: bool = False,
    ) -> Figure:
        """Create eval figure. Returns figure for caller to display/handle."""
        if keys is None:
            keys = ["map", "map_50", "map_75", "mar_100"]
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

        eval_log = self.meta["eval_log"]
        epoch_slice = slice(*epoch_range) if epoch_range is not None else slice(None)
        epochs = list(range(1, len(eval_log) + 1))[epoch_slice]
        eval_log = eval_log[epoch_slice]

        for key in keys:
            ax.plot(
                epochs,
                [e[key] for e in eval_log],
                label=key,
            )
        ax.set_title(f"Evals {label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value")
        ax.legend()
        plt.close(fig)
        if show:
            display(fig)  # type: ignore
        return fig

    def plot_lr(
        self,
        figsize: tuple[int, int] = (12, 3),
        label: str = "",
        epoch_range: tuple[int | None, int | None] | None = None,
        show: bool = False,
    ) -> Figure:
        """Create lr figure. Returns figure for caller to display/handle."""
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

        lr_log = self.meta["lr_log"]
        epoch_slice = slice(*epoch_range) if epoch_range is not None else slice(None)
        epochs = list(range(1, len(lr_log) + 1))[epoch_slice]
        lr_log = lr_log[epoch_slice]

        ax.plot(epochs, lr_log)
        ax.set_title(f"Learning Rate {label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        plt.close(fig)
        if show:
            display(fig)  # type: ignore
        return fig


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
