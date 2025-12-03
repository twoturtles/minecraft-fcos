"""Bounding Boxes"""

import logging
from pathlib import Path
from typing import Annotated, Callable, Iterator, Self, Sequence

import ipywidgets as widgets  # type: ignore
import pandas as pd
from IPython.display import display
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

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

    def create_image_result(
        self, file: str, bboxes: list[BBox]
    ) -> ImageResult:
        """Create an ImageResult with this dataset's base_path pre-set."""
        return ImageResult(file=file, bboxes=bboxes, base_path=self.base_path)


# BBox utils


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
