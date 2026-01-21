"""Bounding Boxes"""

import copy
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, Self, Sequence

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

    def to_df(self, *, size: tuple[int, int] | None = None) -> pd.DataFrame:
        def _coords(b: BBox) -> list[float]:
            return b.xyxyn if size is None else xyxyn_to_xyxy(b.xyxyn, size)

        df = pd.DataFrame(
            [[bbox.category, *_coords(bbox)] for bbox in self.bboxes],
            columns=["category", "x1", "y1", "x2", "y2"],
        )
        return df

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        file: str,
        *,
        size: tuple[int, int] | None = None,
        base_path: Path | str = Path("."),
    ) -> Self:
        """Create ImageResult from DataFrame.
        Expects columns: category, x1, y2, x2, y2. Other columns are ignored"""

        def _coords(coord_in: list[float]) -> list[float]:
            return coord_in if size is None else xyxy_to_xyxyn(coord_in, size)

        bboxes = [
            BBox(
                category=row["category"],
                xyxyn=_coords([row["x1"], row["y1"], row["x2"], row["y2"]]),
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
    def new(
        cls,
        *,
        dset_dir: Path | str,
        categories: list[str],
        input_images_dir: Path | str | None = None,
        info_fname: str = DEFAULT_INFO_FNAME,
        images_sub: str = "images",
        glob_pat: str = "*.png",
    ) -> Self:
        dset_dir = Path(dset_dir)
        dset_dir.mkdir(parents=True)  # Fail if it exists
        info_path = dset_dir / info_fname
        images_dir = dset_dir / images_sub
        images_dir.mkdir()

        dset = cls(categories=categories, file_path=info_path)

        if input_images_dir is not None:
            input_images_dir = Path(input_images_dir)
            image_files = input_images_dir.glob(glob_pat)
            for ifile in image_files:
                shutil.copy2(ifile, images_dir)
                ir = dset.create_image_result(
                    file=f"{images_sub}/{ifile.name}", bboxes=[]
                )
                dset.images.append(ir)

        dset.save()
        print(f"Created {info_path}, {len(dset.images)} images")
        return dset

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
        return pd.DataFrame(
            rows, columns=["image_idx", "file", "category", "x1", "y1", "x2", "y2"]
        )

    def view(self) -> None:
        InferViewer[ImageResult](
            lambda ir: ir, self.images, self.categories
        ).show_widget()

    def dataset_stats(self) -> pd.Series:
        """Calculate useful statistics from a dataset DataFrame.
        Returns: pd.Series with dataset statistics
        """
        df = self.to_df()
        stats: dict[str, Any] = {}

        # Basic counts
        stats["num_images"] = df["image_idx"].nunique()
        stats["num_bboxes"] = len(df)
        stats["avg_bboxes_per_image"] = (
            len(df) / df["image_idx"].nunique() if len(df) > 0 else 0.0
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

    def to_hf_imagefolder(self, output_dir: Path | str) -> None:
        """Export dataset to HuggingFace ImageFolder format with metadata.jsonl.

        Output structure:
            output_dir/
            ├── images/
            │   ├── 001.png
            │   └── ...
            └── metadata.jsonl

        Bounding boxes are absolute xyxy coordinates.
        Load with: load_dataset("imagefolder", data_dir="output_dir")
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_dir / "metadata.jsonl"
        with open(metadata_path, "w") as f:
            for image_result in self.images:
                src_path = image_result.full_path
                dst_name = Path(image_result.file).name
                dst_path = images_dir / dst_name
                shutil.copy2(src_path, dst_path)

                with Image.open(src_path) as img:
                    size = img.size

                metadata = {
                    "file_name": f"images/{dst_name}",
                    "objects": {
                        "bbox": [
                            xyxyn_to_xyxy(bbox.xyxyn, size)
                            for bbox in image_result.bboxes
                        ],
                        "category": [bbox.category for bbox in image_result.bboxes],
                    },
                }
                f.write(json.dumps(metadata) + "\n")

        LOG.info(
            f"Exported {len(self.images)} images to ImageFolder format at {output_dir}"
        )

    def to_coco(self, output_dir: Path | str, add_background: bool = True) -> None:
        """Export dataset to COCO format.

        Output structure:
            output_dir/
            ├── images/
            │   ├── 001.png
            │   └── ...
            └── annotations.json

        Bounding boxes are [x, y, width, height] absolute coordinates.
        add_background - make a __background__ class at id 0 by shifting ids up
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        categories = (
            ["__background__"] + self.categories if add_background else self.categories
        )
        cat_name2id = {name: i for i, name in enumerate(categories)}

        coco: dict[str, Any] = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for name, i in cat_name2id.items()],
        }

        annotation_id = 0
        for image_id, image_result in enumerate(self.images):
            src_path = image_result.full_path
            dst_name = Path(image_result.file).name
            dst_path = images_dir / dst_name
            shutil.copy2(src_path, dst_path)

            with Image.open(src_path) as img:
                width, height = img.size

            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": dst_name,
                    "width": width,
                    "height": height,
                }
            )

            for bbox in image_result.bboxes:
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": cat_name2id[bbox.category],
                        "bbox": xyxyn_to_coco(bbox.xyxyn, (width, height)),
                    }
                )
                annotation_id += 1

        with open(output_dir / "annotations.json", "w") as f:
            json.dump(coco, f, indent=2)

        LOG.info(f"Exported {len(self.images)} images to COCO format at {output_dir}")


#
# Torch dataset
#


class TorchDataset(tv.datasets.VisionDataset):  # type: ignore
    def __init__(
        self,
        root: str | Path,
        info_fname: str = DEFAULT_INFO_FNAME,
        transform: Callable[..., Any] | None = None,
    ):
        super().__init__(root=root, transform=transform)

        self.dset = Dataset.load(Path(root) / info_fname)
        self.category2idx = {name: i for i, name in enumerate(self.dset.categories)}

    @property
    def categories(self) -> list[str]:
        return self.dset.categories

    def __len__(self) -> int:
        return len(self.dset.images)

    def __getitem__(self, idx: int) -> tuple[tv_tensors.Image, dict[str, Any]]:
        img_result = self.dset.images[idx]
        pil_img = Image.open(img_result.full_path).convert("RGB")
        img = tv_tensors.Image(pil_img)

        # Build target dict
        h, w = img.shape[1:]
        boxes_data = [xyxyn_to_xyxy(bb.xyxyn, (w, h)) for bb in img_result.bboxes]
        boxes = tv_tensors.BoundingBoxes(
            boxes_data if boxes_data else torch.zeros((0, 4)),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        )
        labels = torch.tensor(
            [self.category2idx[bbox.category] for bbox in img_result.bboxes],
            dtype=torch.int64,
        )

        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
            "file": img_result.file,
            "image_result": img_result,
        }

        return img, target

    @staticmethod
    def collate_fn(
        batch: list[tuple[tv_tensors.Image, dict[str, Any]]],
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """For use with Dataloader - keep targets as a list"""
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return images, targets


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

        # Internal torch coco-format dataset
        self.coco_dataset = tv.datasets.wrap_dataset_for_transforms_v2(
            # The transforms can be v2 since they're handled by the wrapper.
            tv.datasets.CocoDetection(
                root / images_subdir, root / ann_fname, transforms=v2.ToImage()
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

    def __getitem__(self, idx: int) -> tuple[tv_tensors.Image, dict[str, Any]]:
        item: tuple[tv_tensors.Image, dict[str, Any]] = self.coco_dataset[idx]
        return item

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


class MCDataLoader(torch.utils.data.DataLoader[Any]):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("collate_fn", MCDataset.collate_fn)
        super().__init__(*args, **kwargs)


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
    Plot bounding boxes on copy of img
    Optionally pass categories for consistent colors
    """
    img = img.copy()
    plot_bb_inplace(img, bboxes, categories)
    return img


def plot_bb_inplace(
    img: Image.Image, bboxes: Sequence[BBox], categories: Sequence[str] | None
) -> None:
    """
    Plot bounding boxes on img
    Optionally pass categories for consistent colors
    """
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


def torch_plot_bb(
    img: tv_tensors.Image,
    target: dict[str, Any],
    categories: list[str],
    return_pil: bool = False,
) -> torch.Tensor | Image.Image:
    labels = [categories[i] for i in target["labels"]]
    color_set = tt.Colors().get_rgb()
    color_map = {categories[i]: color_set[i] for i in range(len(categories))}
    colors = [color_map[label] for label in labels]
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
    targets: list[dict[str, Any]],
    categories: list[str],
    nrow: int = 4,
) -> Image.Image:
    result = tv.utils.make_grid(
        [
            torch_plot_bb(img, target, categories)
            for img, target in zip(images, targets)
        ],
        nrow=nrow,
    )
    img: Image.Image = v2.functional.to_pil_image(result)
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


def yr_to_bb(yolo: Results) -> list[BBox]:
    """Convert ultralytics yolo Results to BBoxes"""
    bboxes = []
    if yolo.boxes is not None and len(yolo.boxes) > 0:
        # Get normalized xyxy coordinates
        xyxyn = yolo.boxes.xyxyn.cpu().numpy()  # type: ignore
        # Get class indices
        cls = yolo.boxes.cls.cpu().numpy()  # type: ignore

        for i in range(len(yolo.boxes)):
            class_id = int(cls[i])
            category = yolo.names[class_id]
            coords = xyxyn[i].tolist()

            bboxes.append(BBox(category=category, xyxyn=coords))

    return bboxes


def yr_to_ir(yolo: Results) -> ImageResult:
    """Convert ultralytics yolo Results to ImageResult"""
    bboxes = yr_to_bb(yolo)
    file_path = str(yolo.path) if yolo.path else ""
    return ImageResult(file=file_path, bboxes=bboxes)
