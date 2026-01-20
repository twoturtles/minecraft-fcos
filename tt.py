"""utils"""

import logging
from pathlib import Path
from typing import TypeVar

import ipywidgets as widgets  # type: ignore
import pandas as pd
import torch
from ipydatagrid import DataGrid  # type: ignore
from IPython.display import display
from jupyter_bbox_widget import BBoxWidget  # type: ignore
from PIL import Image
from torch.utils.data import Dataset, random_split

LOG = logging.getLogger(__name__)

##
# Logging

# _FMT = "[%(asctime)s] [%(threadName)s/%(levelname)s] (%(name)s) %(message)s"
_FMT = "[%(asctime)s]:[%(levelname)s]:(%(name)s): %(message)s"
_DATEFMT = "%H:%M:%S"


def logging_init(
    *,
    level: int = logging.INFO,
    use_colors: bool = True,
) -> None:
    if use_colors:
        handler = logging.StreamHandler()
        handler.setFormatter(LogColorFormatter())
        logging.basicConfig(level=level, handlers=[handler])
    else:
        logging.basicConfig(level=level, format=_FMT, datefmt=_DATEFMT)


class LogColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[41m",  # Red background
    }

    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, "")
        fmt = f"{color}{_FMT}{self.RESET}"
        formatter = logging.Formatter(fmt, datefmt=_DATEFMT)
        return formatter.format(record)


##
# Torch

_T = TypeVar("_T")


def seed(seed: int) -> None:
    """Set global seeds"""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_torch_device(print_device: bool = True) -> str:
    """Get cpu, gpu, or mps device"""
    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Torch device: {device}")

    return device


def split1(
    dataset: Dataset[_T], train_pct: float, seed: int
) -> tuple[Dataset[_T], Dataset[_T]]:
    """Simple wrap for a single split"""
    len_dset = len(dataset)  # type: ignore[arg-type] # Dataset has __len__ at runtime
    n_train = int(len_dset * train_pct)
    n_valid = len_dset - n_train

    train_data, valid_data = random_split(
        dataset,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_data, valid_data


##
# Viewers


class ImageDirViewer:
    def __init__(self, image_dir: str | Path, glob_pat: str = "*.png"):
        self.image_dir = image_dir
        self.image_files = sorted(Path(image_dir).glob(glob_pat))
        self.current_file = None
        self.current_index = 0

    def view_image_cb(self, index: int) -> None:
        self.current_index = index
        self.current_file = self.image_files[index]

        img = Image.open(self.current_file)
        print(f"dir={self.image_dir} n_images={len(self.image_files)}")
        print(f"index={index} file={self.current_file.name}")
        display(img)

    def show_widget(self) -> None:
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.image_files) - 1,
            description="Image:",
            continuous_update=False,
        )
        widgets.interact(self.view_image_cb, index=slider)


##
# Colors


class Colors:
    schemes = dict(
        # https://github.com/d3/d3-scale-chromatic/blob/main/src/categorical/category10.js
        category10="1f77b4ff7f0e2ca02cd627289467bd8c564be377c27f7f7fbcbd2217becf",
        # https://github.com/vega/vega/blob/main/packages/vega-scale/src/palettes.js
        category20="1f77b4aec7e8ff7f0effbb782ca02c98df8ad62728ff98969467bdc5b0d58c564bc49c94e377c2f7b6d27f7f7fc7c7c7bcbd22dbdb8d17becf9edae5",
        category20b="393b795254a36b6ecf9c9ede6379398ca252b5cf6bcedb9c8c6d31bd9e39e7ba52e7cb94843c39ad494ad6616be7969c7b4173a55194ce6dbdde9ed6",
        category20c="3182bd6baed69ecae1c6dbefe6550dfd8d3cfdae6bfdd0a231a35474c476a1d99bc7e9c0756bb19e9ac8bcbddcdadaeb636363969696bdbdbdd9d9d9",
    )

    def get_rgb(self, scheme: str = "category20") -> list[tuple[int, int, int]]:
        """Get a color scheme as a list of RGB tuples."""
        # Convert each hex color to RGB tuple
        rgb_colors: list[tuple[int, int, int]] = []
        for h in self.hex_split(scheme):
            rgb_colors.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
        return rgb_colors

    def get_strs(self, scheme: str = "category20") -> list[str]:
        hex_list = self.hex_split(scheme)
        return [f"#{color}" for color in hex_list]

    def hex_split(self, scheme: str) -> list[str]:
        # Split into 6-character chunks
        hex_string = self.schemes[scheme]
        return [hex_string[i : i + 6] for i in range(0, len(hex_string), 6)]
