"""Bounding Boxes"""

import copy
import logging
import random
import shutil
from pathlib import Path
from typing import Annotated, Callable, Iterator, Self, Sequence

import ipywidgets as widgets  # type: ignore
import pandas as pd
from IPython.display import display
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from ruamel.yaml import YAML

import tt

LOG = logging.getLogger(__name__)


class BBox(BaseModel):
    category: str
    xyxyn: Annotated[list[float], Field(min_length=4, max_length=4)]


class ImageResult(BaseModel):
    file: str  # Relative to base_path
    bboxes: list[BBox]
    base_path: Annotated[Path | None, Field(exclude=True)] = None

    @property
    def full_path(self) -> Path:
        if self.base_path:
            return self.base_path / self.file
        else:
            return Path(self.file)

    def plot_bb(self, categories: list[str] | None = None) -> Image.Image:
        return plot_bb(Image.open(self.full_path), self.bboxes, categories)

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [[bbox.category, *bbox.xyxyn] for bbox in self.bboxes],
            columns=["category", "x1", "y1", "x2", "y2"],
        )
        return df

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, file: str, base_path: Path | str = Path(".")
    ) -> Self:
        """Create ImageResult from DataFrame"""
        bboxes = [
            BBox(
                category=row["category"],
                xyxyn=[row["x1"], row["y1"], row["x2"], row["y2"]],
            )
            for idx, row in df.iterrows()
        ]
        return cls(file=file, bboxes=bboxes, base_path=base_path)


class Dataset(BaseModel):
    categories: list[str] = []  # Pydantic can handle mutable defaults
    images: list[ImageResult] = []
    file_path: Annotated[Path | None, Field(exclude=True)] = None

    @property
    def base_path(self) -> Path:
        if self.file_path:
            return self.file_path.parent
        else:
            return Path(".")

    def save(self, file_path: Path | str | None = None) -> None:
        if file_path is None:
            file_path = self.file_path
            assert file_path is not None
        file_path = Path(file_path)
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, file_path: Path | str) -> Self:
        file_path = Path(file_path)
        with open(file_path, "r") as f:
            json_str = f.read()
        dset = cls.model_validate_json(json_str)
        dset.file_path = file_path
        for ir in dset.iter_images():
            ir.base_path = dset.base_path
        return dset

    def copy_deep(self) -> Self:
        return self.model_copy(deep=True)

    def iter_images(self) -> Iterator[ImageResult]:
        for image in self.images:
            yield image

    def create_image_result(self, file: str, bboxes: list[BBox]) -> ImageResult:
        """Create an ImageResult with this dataset's base_path pre-set."""
        return ImageResult(file=file, bboxes=bboxes, base_path=self.base_path)

    def to_df(self) -> pd.DataFrame:
        """Convert dataset to DataFrame"""
        rows = []
        for idx, image_result in enumerate(self.iter_images()):
            full_path = str(image_result.full_path)
            for bbox in image_result.bboxes:
                rows.append(
                    {
                        "image_idx": idx,
                        "file": full_path,
                        "category": bbox.category,
                        "x1": bbox.xyxyn[0],
                        "y1": bbox.xyxyn[1],
                        "x2": bbox.xyxyn[2],
                        "y2": bbox.xyxyn[3],
                    }
                )
        return pd.DataFrame(rows)

    def dataset_stats(self) -> pd.Series:
        """Calculate useful statistics from a dataset DataFrame.
        Returns: pd.Series with dataset statistics
        """
        df = self.to_df()
        stats = {}

        # Basic counts
        stats["num_images"] = df["image_idx"].nunique()
        stats["num_bboxes"] = len(df)
        stats["avg_bboxes_per_image"] = (
            len(df) / df["image_idx"].nunique() if len(df) > 0 else 0
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

        # Bounding box size statistics (normalized coordinates)
        if len(df) > 0:
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

    def to_yolo(
        self,
        output_dir: Path | str,
        train_pct: int | float,
        seed: int | float | None = None,
    ) -> None:
        """Export dataset to YOLO format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_data: dict[str, str | dict[int, str]] = {
            "path": str(output_dir.absolute()),
        }

        train_irs, val_irs = split_ir(self.images, train_pct, seed)
        for split_name, split_irs in [("train", train_irs), ("val", val_irs)]:
            images_sub = f"images/{split_name}"
            labels_sub = f"labels/{split_name}"
            image_results_to_yolo(
                split_irs,
                self.categories,
                output_dir / images_sub,
                output_dir / labels_sub,
            )
            yaml_data[split_name] = images_sub

        # Write data.yaml
        yaml_data["names"] = {i: category for i, category in enumerate(self.categories)}
        yaml = YAML()
        with open(output_dir / "data.yaml", "w") as f:
            yaml.dump(yaml_data, f)

        LOG.info(f"Exported {len(self.images)} images to YOLO format at {output_dir}")


# BBox utils


def image_results_to_yolo(
    image_results: list[ImageResult],
    categories: list[str],
    images_dir: Path,
    labels_dir: Path,
) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    cat_map = {cat: i for i, cat in enumerate(categories)}

    for image_result in image_results:
        # Copy image file
        src_img = image_result.full_path
        dst_img = images_dir / Path(image_result.file).name
        shutil.copy2(src_img, dst_img)

        # Write YOLO annotations
        label_file = labels_dir / (Path(image_result.file).stem + ".txt")
        with open(label_file, "w") as f:
            for bbox in image_result.bboxes:
                class_id = cat_map[bbox.category]
                x, y, w, h = xyxyn_to_yolo(bbox.xyxyn)
                f.write(f"{class_id} {x} {y} {w} {h}\n")


def split_ir(
    images: list[ImageResult], pct1: float | int, seed: int | float | None = None
) -> tuple[list[ImageResult], list[ImageResult]]:
    """Split list of images based on passed percentage"""
    pct = pct1 / 100.0 if pct1 >= 1 else float(pct1)
    pct = max(0.0, min(pct, 1.0))
    n1 = int(pct * len(images))
    rnd = random if seed is None else random.Random(seed)
    shuffled = copy.deepcopy(images)
    rnd.shuffle(shuffled)
    return shuffled[:n1], shuffled[n1:]


def sort_xyxy[T: (int, float)](xyxy: list[T]) -> list[T]:
    """Sort corners to ensure x1 < x2 and y1 < y2."""
    x1, y1, x2, y2 = xyxy
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def clamp[T: (int, float)](val: T, min_val: T, max_val: T) -> T:
    return max(min_val, min(val, max_val))


def xyxyn_to_xyxy(xyxyn: list[float], size: tuple[int, int]) -> list[float]:
    """Convert normalized coords to absolute. size is (width, height)"""
    width, height = size
    xyxy = [
        xyxyn[0] * width,
        xyxyn[1] * height,
        xyxyn[2] * width,
        xyxyn[3] * height,
    ]
    return sort_xyxy(xyxy)


def xyxy_to_xyxyn(xyxy: list[float], size: tuple[int, int]) -> list[float]:
    width, height = size
    xyxyn = [
        xyxy[0] / width,
        xyxy[1] / height,
        xyxy[2] / width,
        xyxy[3] / height,
    ]
    return sort_xyxy(xyxyn)


def xyxy_to_coco(xyxy: list[float]) -> list[float]:
    # coco is [x, y, width, height] absolute
    return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]


def coco_to_xyxy(coco: list[float]) -> list[float]:
    return [coco[0], coco[1], coco[2] + coco[0], coco[3] + coco[1]]


def xyxyn_to_coco(xyxyn: list[float], size: tuple[int, int]) -> list[float]:
    return xyxy_to_coco(xyxyn_to_xyxy(xyxyn, size))


def coco_to_xyxyn(coco: list[float], size: tuple[int, int]) -> list[float]:
    return xyxy_to_xyxyn(coco_to_xyxy(coco), size)


def xyxyn_to_yolo(xyxyn: list[float]) -> list[float]:
    # yolo is (center_x, center_y, width, height) normalized
    x1, y1, x2, y2 = xyxyn
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return [center_x, center_y, width, height]


def plot_bb(
    img: Image.Image, bboxes: Sequence[BBox], categories: Sequence[str] | None
) -> Image.Image:
    """
    Plot bounding boxes
    Optionally pass categories for consistent colors
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    if categories is None:
        # Use labels to create categories set.
        categories = sorted(list(set([bbox.category for bbox in bboxes])))

    colors = tt.Colors().get_rgb()
    color_map = {categories[i]: colors[i] for i in range(len(categories))}
    for bbox in bboxes:
        color = color_map[bbox.category]
        xyxy = xyxyn_to_xyxy(bbox.xyxyn, img.size)
        draw.rectangle(((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])), outline=color, width=2)
        draw.text((xyxy[0] + 4, xyxy[1] + 2), bbox.category, fill=color, font_size=16)

    return img


class InferViewer[T]:
    """Pass list and function that turns a list item into an ImageResult."""

    def __init__(
        self,
        infer_fn: Callable[[T], ImageResult],
        infer_list: list[T],
        categories: list[str] | None = None,
    ):
        self.infer_fn = infer_fn
        self.infer_list = infer_list
        self.categories = categories

    def view_image_cb(self, index: int) -> None:
        # Call the provided inference function
        result = self.infer_fn(self.infer_list[index])
        print(f"index={index} file={result.file}")
        display(result.plot_bb(categories=self.categories))  # type: ignore[no-untyped-call]

    def show_widget(self) -> None:
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.infer_list) - 1,
            description="Image:",
            continuous_update=False,
        )
        widgets.interact(self.view_image_cb, index=slider)
