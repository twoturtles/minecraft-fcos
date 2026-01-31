from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv  # type: ignore
from IPython.display import display
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import fcos  # type: ignore
from torchvision.transforms import v2 as v2  # type: ignore
from tqdm.auto import tqdm, trange

import bb


class Detection(TypedDict):
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


class FCOSTrainer:

    def __init__(
        self,
        *,
        categories: list[str],
        project_dir: Path | str,
        load_checkpoint: Path | str | int | None = None,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.4,
        device: str | torch.device = "mps",
    ) -> None:
        self.categories = categories
        self.num_categories = len(categories)
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = self.project_dir.name
        self.device = torch.device(device)
        self.best_checkpoint = self.project_dir / "best.pt"
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        self.preprocess = fcos.FCOS_ResNet50_FPN_Weights.COCO_V1.transforms()

        if load_checkpoint is None:
            self._load_pretrained()
        else:
            ckpt_file = (
                self.project_dir / f"ep-{load_checkpoint}.pt"
                if isinstance(load_checkpoint, int)
                else self.project_dir / str(load_checkpoint)
            )
            self._load_checkpoint(ckpt_file)

    def _load_pretrained(self) -> None:
        """Load FCOS pretrained weights.
        This is expected: missing_keys=[
            "head.classification_head.cls_logits.weight",
            "head.classification_head.cls_logits.bias",
        ],
        """
        print("Initializing new model")
        # FCOS init
        self.model = fcos.fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=self.num_categories,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
        )
        # Load pretrained
        model_state_dict = fcos.FCOS_ResNet50_FPN_Weights.COCO_V1.get_state_dict(
            progress=True, check_hash=True
        )

        # Get rid of classification head weights
        del model_state_dict["head.classification_head.cls_logits.weight"]
        del model_state_dict["head.classification_head.cls_logits.bias"]

        self._setup_model(model_state_dict=model_state_dict)

    def _setup_model(
        self,
        *,
        model_state_dict: dict[str, Any],
        optimizer_state_dict: dict[str, Any] | None = None,
        scheduler_state_dict: dict[str, Any] | None = None,
        total_epochs: int = 0,
        best_map: float = 0.0,
        best_epoch: int = 0,
        loss_log: list[float] | None = None,
        eval_log: list[dict[str, Any]] | None = None,
        lr_log: list[float] | None = None,
    ) -> None:
        self.total_epochs = total_epochs
        self.best_map = best_map
        self.best_epoch = best_epoch
        self.loss_log = loss_log if loss_log else []  # Per-iteration
        self.eval_log = eval_log if eval_log else []  # Per-epoch
        self.lr_log = lr_log if lr_log else []  # Per-epoch

        err_keys = self.model.load_state_dict(model_state_dict, strict=False)
        print(f"err_keys = {err_keys}")
        self.model.to(self.device)
        self._set_requires_grad()

        self.optimizer = torch.optim.AdamW(params=self.model.parameters())
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer, T_0=20
        # )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        if scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)

    def _load_checkpoint(self, ckpt_file: Path | str) -> None:
        ckpt_file = Path(ckpt_file)
        print(f"Loading checkpoint: {ckpt_file}")
        self.model = fcos.fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=self.num_categories,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
        )
        ckpt = torch.load(ckpt_file, weights_only=True)
        self._setup_model(
            model_state_dict=ckpt["model_state_dict"],
            optimizer_state_dict=ckpt["optimizer_state_dict"],
            scheduler_state_dict=ckpt["scheduler_state_dict"],
            total_epochs=ckpt["total_epochs"],
            best_map=ckpt.get("best_map", 0.0),
            best_epoch=ckpt.get("best_epoch", 0),
            loss_log=ckpt["loss_log"],
            eval_log=ckpt.get("eval_log"),
            lr_log=ckpt.get("lr_log"),
        )

    def save_checkpoint(self, ckpt_file: Path | str | None = None) -> None:
        ckpt_file = (
            self.project_dir / f"ep-{self.total_epochs}.pt"
            if ckpt_file is None
            else Path(ckpt_file)
        )
        checkpoint = {
            "total_epochs": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_map": self.best_map,
            "best_epoch": self.best_epoch,
            "loss_log": self.loss_log,
            "eval_log": self.eval_log,
            "lr_log": self.lr_log,
        }
        torch.save(checkpoint, ckpt_file)

    def _set_requires_grad(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.model.head.parameters():
            param.requires_grad_(True)
        for param in self.model.backbone.fpn.parameters():
            param.requires_grad_(True)

    def topk_preds(self, preds: list[Detection], k: int) -> list[Detection]:
        """Filter predictions by top-k."""
        filtered_preds: list[Detection] = []
        for pred in preds:
            boxes, scores, labels = pred["boxes"], pred["scores"], pred["labels"]
            topk_scores, topk_indices = torch.topk(scores, k=min(k, len(scores)))
            filtered_pred: Detection = {
                "boxes": boxes[topk_indices],
                "scores": topk_scores,
                "labels": labels[topk_indices],
            }
            filtered_preds.append(filtered_pred)
        return filtered_preds

    def filter_preds(self, preds: list[Detection], thresh: float) -> list[Detection]:
        """Filter predictions by score threshold."""
        filtered_preds: list[Detection] = []
        for pred in preds:
            boxes, scores, labels = pred["boxes"], pred["scores"], pred["labels"]
            mask = scores >= thresh
            filtered_pred: Detection = {
                "boxes": boxes[mask],
                "scores": scores[mask],
                "labels": labels[mask],
            }
            filtered_preds.append(filtered_pred)
        return filtered_preds

    def infer(self, img: tv.tv_tensors.Image) -> Detection:
        batch = img.unsqueeze(0)
        return self.forward(batch)[0]

    def forward(self, batch: torch.Tensor) -> list[Detection]:
        self.model.eval()
        batch = self.preprocess(batch.to(self.device))
        with torch.inference_mode():
            preds: list[Detection] = self.model(batch)
        return preds

    def plot_infer(
        self, img: tv.tv_tensors.Image, topk: int | None = None
    ) -> Image.Image:
        pred = self.infer(img)
        if topk != None:
            pred = self.topk_preds([pred], k=topk)[0]
        ret = bb.torch_plot_bb(
            img, pred, self.categories, include_scores=True, return_pil=True
        )
        assert isinstance(ret, Image.Image)
        return ret

    def plot_loss(
        self,
        figsize: tuple[int, int] = (12, 3),
        label: str = "",
        epoch_range: tuple[int | None, int | None] | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create loss figure. Returns figure for caller to display/handle."""
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

        if epoch_range is None:
            iter_slice = slice(None)
        else:
            iters_per_epoch = len(self.loss_log) // self.total_epochs
            start = (
                epoch_range[0] * iters_per_epoch if epoch_range[0] is not None else None
            )
            end = (
                epoch_range[1] * iters_per_epoch if epoch_range[1] is not None else None
            )
            iter_slice = slice(start, end)

        train_x = np.linspace(0, self.total_epochs, len(self.loss_log))[iter_slice]
        loss_log = self.loss_log[iter_slice]

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
    ) -> plt.Figure:
        """Create eval figure. Returns figure for caller to display/handle."""
        if keys is None:
            keys = ["map", "map_50", "map_75", "mar_100"]
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

        epoch_slice = slice(*epoch_range) if epoch_range is not None else slice(None)
        epochs = list(range(1, len(self.eval_log) + 1))[epoch_slice]
        eval_log = self.eval_log[epoch_slice]

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
    ) -> plt.Figure:
        """Create lr figure. Returns figure for caller to display/handle."""
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

        epoch_slice = slice(*epoch_range) if epoch_range is not None else slice(None)
        epochs = list(range(1, len(self.lr_log) + 1))[epoch_slice]
        lr_log = self.lr_log[epoch_slice]

        ax.plot(epochs, lr_log)
        ax.set_title(f"Learning Rate {label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        plt.close(fig)
        if show:
            display(fig)  # type: ignore
        return fig

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
            f"Final epochs={self.total_epochs} loss={self.loss_log[-1]:.4f} mAP={self.eval_log[-1]['map']:.4f}"
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
            self.loss_log.append(loss.item())

            # Zero gradients
            self.optimizer.zero_grad()

            # Backpropagate gradients
            loss.backward()
            # Do a single optimization step
            self.optimizer.step()

        self.total_epochs += 1
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.lr_log.append(current_lr)
        self.scheduler.step()

        if val_loader is not None:
            metrics = self.evaluate(val_loader)
            self.eval_log.append(metrics)
            # self.scheduler.step(metrics["map"])  # For ReduceLROnPlateau
            if metrics["map"] > self.best_map:
                self.best_map = metrics["map"]
                self.best_epoch = self.total_epochs
                print(f"New best mAP={self.best_map:.4f} at epoch {self.best_epoch}")
                self.save_checkpoint(self.best_checkpoint)

        print(
            f"Epoch {self.total_epochs}: val mAP={metrics['map']:.4f} lr={current_lr:.6f}"
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
