"""Bounding Boxes and Object Detection Utilities"""

import copy
import json
import logging
import random
import shutil
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Iterator,
    Mapping,
    NamedTuple,
    Self,
    Sequence,
    TypedDict,
    TypeIs,
    TypeVar,
)

import ipywidgets as widgets  # type: ignore
import pandas as pd
import torch
import torchvision as tv  # type: ignore
from IPython.display import display
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from ruamel.yaml import YAML
from torchvision import tv_tensors
from torchvision.transforms import v2  # type: ignore
from ultralytics.engine.results import Results

import tt

LOG = logging.getLogger(__name__)

DEFAULT_INFO_FNAME: str = "info.json"

_T = TypeVar("_T")


class BaseAnnotation(TypedDict):
    """Common annotation fields"""

    boxes: tv_tensors.BoundingBoxes
    labels: torch.Tensor


class Detection(BaseAnnotation):
    """Object Detection Prediction (includes BaseAnnotation)"""

    scores: torch.Tensor


class Target(BaseAnnotation):
    """Dataset Target (includes BaseAnnotation)"""

    image_id: int


class MCDatasetItem(NamedTuple):
    """An entry in a MCDataset"""

    image: tv_tensors.Image
    target: Target


def is_detection(target: Target | Detection) -> TypeIs[Detection]:
    return "scores" in target


class MCDataset(tv.datasets.VisionDataset):  # type: ignore
    """Torch dataset for Minecraft data using COCO format"""

    def __init__(
        self,
        root: str | Path,
        images_subdir: str = "images",
        ann_fname: str = "annotations.json",
        transform: Callable[..., Any] | None = None,
    ):
        root = Path(root)
        super().__init__(root=root, transform=transform)
        self.ann_path = root / ann_fname
        self.images_path = root / images_subdir

        # Internal torch coco-format dataset
        self.coco_dataset = tv.datasets.wrap_dataset_for_transforms_v2(
            # The transforms can be v2 since they're handled by the wrapper.
            tv.datasets.CocoDetection(
                self.images_path, self.ann_path, transforms=v2.ToImage()
            )
        )

        # dict[id, category name]
        self.id2category: dict[int, str] = {
            cat["id"]: cat["name"]
            for cat in self.coco_dataset.coco.loadCats(
                self.coco_dataset.coco.getCatIds()
            )
        }
        self.categories = list(self.id2category.values())

    def __len__(self) -> int:
        return len(self.coco_dataset)

    def __getitem__(self, idx: int) -> MCDatasetItem:
        item: MCDatasetItem = self.coco_dataset[idx]
        image, target = item

        # Handle images with no boxes
        if "boxes" not in target:
            h, w = image.shape[1:]
            target["boxes"] = tv_tensors.BoundingBoxes(
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            )

        if "labels" not in target:
            target["labels"] = torch.zeros(0, dtype=torch.int64)

        if self.transform:
            transformed: MCDatasetItem = self.transform(image, target)
            image, target = transformed

        ret = MCDatasetItem(image, target)
        return ret

    def view(self) -> None:
        BBoxViewer(self).show_widget()

    def image_info(self, idx: int) -> dict[str, Any]:
        """Return the coco info for an image
        {"id": 3, "file_name": "frame_208322.png",
         "width": 640, "height": 640},
        """
        info: dict[str, Any] = self.coco_dataset.coco.loadImgs([idx])[0]
        return info

    def image_path(self, idx: int) -> Path:
        """Return the image absolute path"""
        return Path(self.images_path / self.image_info(idx)["file_name"]).absolute()

    def add_annotation(self, idx: int, ann: BaseAnnotation) -> None:
        """Add annotations to an image (replaces existing).
        NOTE: You must run rebuild_index() after completing updates to have the
        changes reflected in the values returned by dataset indexing.
        """
        coco = self.coco_dataset.coco
        image_id = coco.dataset["images"][idx]["id"]

        # Remove existing annotations for this image
        coco.dataset["annotations"] = [
            ann for ann in coco.dataset["annotations"] if ann["image_id"] != image_id
        ]

        # Get next annotation id
        if coco.dataset["annotations"]:
            next_ann_id = max(ann["id"] for ann in coco.dataset["annotations"]) + 1
        else:
            next_ann_id = 0

        # Convert to XYWH (COCO format) and add annotations
        boxes_xywh = v2.functional.convert_bounding_box_format(
            ann["boxes"], new_format=tv_tensors.BoundingBoxFormat.XYWH
        ).tolist()
        labels = ann["labels"]
        for i in range(len(boxes_xywh)):
            coco_ann = {
                "id": next_ann_id + i,
                "image_id": image_id,
                "category_id": int(labels[i].item()),
                "bbox": boxes_xywh,
                "area": boxes_xywh[2] * boxes_xywh[3],
                "iscrowd": 0,
            }
            coco.dataset["annotations"].append(coco_ann)

    def rebuild_index(self) -> None:
        """Rebuild indices in the underlying pycocotools. This updates the
        caches used for indexing"""
        self.coco_dataset.coco.createIndex()

    def save_annotations(self, ann_path: Path | str | None = None) -> None:
        """Save annotations to disk."""
        ann_path = Path(ann_path) if ann_path else self.ann_path
        with open(self.ann_path, "w") as f:
            json.dump(self.coco_dataset.coco.dataset, f, indent=2)

    @classmethod
    def new(
        cls,
        *,
        dset_dir: Path | str,
        categories: list[str],
        input_images_dir: Path | str | None = None,
        images_subdir: str = "images",
        ann_fname: str = "annotations.json",
        glob_pat: str = "*.png",
    ) -> Self:
        dset_dir = Path(dset_dir)
        dset_dir.mkdir(parents=True)  # Fail if it exists
        images_path = dset_dir / images_subdir
        images_path.mkdir()

        coco: dict[str, Any] = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": id, "name": name} for id, name in enumerate(categories)
            ],
        }

        if input_images_dir is not None:
            input_images_dir = Path(input_images_dir)
            image_files = input_images_dir.glob(glob_pat)
            for image_id, image_file in enumerate(image_files):
                shutil.copy2(image_file, images_path)
                with Image.open(image_file) as img:
                    width, height = img.size
                coco["images"].append(
                    {
                        "id": image_id,
                        "file_name": image_file.name,
                        "width": width,
                        "height": height,
                    }
                )

        with open(dset_dir / ann_fname, "w") as f:
            json.dump(coco, f, indent=2)

        dset = cls(dset_dir, images_subdir=images_subdir, ann_fname=ann_fname)
        print(f"Created {dset.root}, {len(dset)} images")
        return dset

    # https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html
    # We need a custom collation function here, since the object detection
    # models expect a sequence of images and target dictionaries. The default
    # collation function tries to torch.stack() the individual elements,
    # which fails in general for object detection, because the number of bounding
    # boxes varies between the images of the same batch.
    # Alternative:
    # collate_fn=lambda batch: tuple(zip(*batch))
    @staticmethod
    def collate_fn(
        batch: list[tuple[tv_tensors.Image, dict[str, Any]]],
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """For use with Dataloader - keep targets as a list"""
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return images, targets

    def to_df(self) -> pd.DataFrame:
        """Convert dataset to DataFrame with absolute xyxy coords.
        Images without annotations get a row with None for bbox fields.
        """
        coco = self.coco_dataset.coco
        rows = []
        for img_idx, img_id in enumerate(coco.getImgIds()):
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info["file_name"]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

            if not anns:
                rows.append(
                    {
                        "image_idx": img_idx,
                        "file": file_name,
                        "category": None,
                        "x1": None,
                        "y1": None,
                        "x2": None,
                        "y2": None,
                    }
                )
            else:
                for ann in anns:
                    xyxy = coco_to_xyxy(ann["bbox"])
                    rows.append(
                        {
                            "image_idx": img_idx,
                            "file": file_name,
                            "category": self.id2category[ann["category_id"]],
                            "x1": xyxy[0],
                            "y1": xyxy[1],
                            "x2": xyxy[2],
                            "y2": xyxy[3],
                        }
                    )
        return pd.DataFrame(
            rows, columns=["image_idx", "file", "category", "x1", "y1", "x2", "y2"]
        )

    def dataset_stats(self) -> pd.Series:
        """Calculate useful statistics from dataset.
        Returns: pd.Series with dataset statistics
        """
        df = self.to_df()
        stats: dict[str, Any] = {}

        # Basic counts
        stats["num_images"] = df["image_idx"].nunique()
        stats["num_bboxes"] = df["category"].notna().sum()
        stats["avg_bboxes_per_image"] = (
            stats["num_bboxes"] / stats["num_images"]
            if stats["num_images"] > 0
            else 0.0
        )

        # Category statistics
        stats["num_categories"] = df["category"].nunique()
        category_counts = df["category"].value_counts()
        stats["most_common_category"] = (
            category_counts.index[0] if len(category_counts) > 0 else None
        )
        stats["least_common_category"] = (
            category_counts.index[-1] if len(category_counts) > 0 else None
        )

        # Bounding box size statistics (absolute coordinates)
        if stats["num_bboxes"] > 0:
            df["width"] = df["x2"] - df["x1"]
            df["height"] = df["y2"] - df["y1"]
            df["area"] = df["width"] * df["height"]

            stats["avg_bbox_width"] = df["width"].mean()
            stats["avg_bbox_height"] = df["height"].mean()
            stats["avg_bbox_area"] = df["area"].mean()
            stats["min_bbox_area"] = df["area"].min()
            stats["max_bbox_area"] = df["area"].max()

        return pd.Series(stats)

    def category_stats(self) -> pd.DataFrame:
        """Get detailed statistics per category.
        Returns: DataFrame with per-category statistics
        """
        df = self.to_df()
        if len(df) == 0:
            return pd.DataFrame()

        df["width"] = df["x2"] - df["x1"]
        df["height"] = df["y2"] - df["y1"]
        df["area"] = df["width"] * df["height"]

        stats = (
            df.groupby("category")
            .agg(
                count=("category", "size"),
                avg_width=("width", "mean"),
                avg_height=("height", "mean"),
                avg_area=("area", "mean"),
            )
            .round(4)
        )

        return stats.sort_values("count", ascending=False)


class TransformedSubset(torch.utils.data.Dataset[_T]):
    """Dataset wrapper to apply transforms to a subset. Needed since torch
    random_split returns Subset objects which apply the transforms of the
    original. Usage - first do split, then wrap the subset that needs transforms
    with this class"""

    def __init__(
        self, subset: torch.utils.data.Subset[_T], transform: Callable[[_T], _T]
    ) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int) -> _T:
        item = self.subset[idx]
        return self.transform(item)

    def __len__(self) -> int:
        return len(self.subset)


def torch_plot_bb(
    img: tv_tensors.Image,
    target: Target | Detection,
    categories: list[str],
    include_scores: bool = False,
    return_pil: bool = False,
) -> torch.Tensor | Image.Image:
    cat_ids = target["labels"].tolist()
    if include_scores and is_detection(target):
        labels = [
            f"{categories[cat_ix]} {score:.2f}"
            for cat_ix, score in zip(cat_ids, target["scores"])
        ]
    else:
        labels = [categories[cat_ix] for cat_ix in cat_ids]
    color_set = tt.Colors().get_rgb()
    cat_ix2color = {cat_ix: color_set[cat_ix] for cat_ix in range(len(categories))}
    colors = [cat_ix2color[cat_ix] for cat_ix in cat_ids]
    result: torch.Tensor = tv.utils.draw_bounding_boxes(
        img,
        boxes=target["boxes"],
        labels=labels,
        colors=colors,
        width=2,
        font="/System/Library/Fonts/Helvetica.ttc",  # macOS
        font_size=20,
    )
    if return_pil:
        pil_img: Image.Image = v2.functional.to_pil_image(result)
        return pil_img
    return result


def plot_bb_grid(
    images: list[tv_tensors.Image],
    targets: list[Target | Detection],
    categories: list[str],
    nrow: int = 4,
    include_scores: bool = False,
) -> Image.Image:
    result = tv.utils.make_grid(
        [
            torch_plot_bb(img, target, categories, include_scores=include_scores)
            for img, target in zip(images, targets)
        ],
        nrow=nrow,
    )
    img: Image.Image = v2.functional.to_pil_image(result)
    return img


class BBoxViewer:
    """Pass dataset and function that turns a tv_tensors.Image into a Detection
    If no infer_fn is passed, the target from the dataset is used."""

    def __init__(
        self,
        dset: MCDataset,
        *,
        infer_fn: Callable[[tv_tensors.Image], Detection] | None = None,
    ):
        self.dset = dset
        self.infer_fn = infer_fn

    def view_image_cb(self, index: int) -> None:
        # Call the provided inference function
        image, target = self.dset[index]
        info = self.dset.image_info(index)
        results = self.infer_fn(image) if self.infer_fn else target
        print(f"index={index} file={info["file_name"]}")
        display(  # type: ignore[no-untyped-call]
            torch_plot_bb(
                image,
                results,
                self.dset.categories,
                include_scores=True,
                return_pil=True,
            )
        )

    def show_widget(self) -> None:
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.dset) - 1,
            description="Image:",
            continuous_update=False,
        )
        widgets.interact(self.view_image_cb, index=slider)
