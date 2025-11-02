"""utils"""

import logging
from pathlib import Path
from typing import Any, Callable, Sequence, TypedDict

import ipywidgets as widgets  # type: ignore
import pandas as pd
from IPython.display import display  # type: ignore
from PIL import Image, ImageColor, ImageDraw

LOG = logging.getLogger(__name__)

##
# Logging


def logging_add_arg(
    parser: argparse.ArgumentParser, default: int | str = "INFO"
) -> None:
    """Add a default logging argument to argparse"""
    parser.add_argument(
        "--log-level",
        "-L",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=default,
        help="Set the logging level (default: INFO)",
    )


_FMT = "[%(asctime)s] [%(threadName)s/%(levelname)s] (%(name)s) %(message)s"
_DATEFMT = "%H:%M:%S"


def logging_init(
    *,
    args: argparse.Namespace | None = None,
    level: int | str | None = None,
    use_colors: bool = True,
) -> None:
    """Default log init. If args are passed (see logging_add_arg), level is pulled
    from that. Otherwise uses a passed in level. Finally defaults to INFO"""
    if args is not None:
        level = getattr(logging, args.log_level.upper(), logging.INFO)
    elif level is None:
        level = logging.INFO

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


class ImageDirViewer:
    def __init__(self, image_dir: str | Path, glob_pat: str = "*.png"):
        self.image_dir = image_dir
        self.image_files = sorted(Path(image_dir).glob(glob_pat))
        self.current_file = None
        self.current_index = 0

    def view_image_cb(self, index: int):
        self.current_index = index
        self.current_file = self.image_files[index]

        img = Image.open(self.current_file)
        print(f"dir={self.image_dir} n_images={len(self.image_files)}")
        print(f"index={index} file={self.current_file.name}")
        display(img)

    def show_widget(self):
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.image_files) - 1,
            description="Image:",
            continuous_update=False,
        )
        widgets.interact(self.view_image_cb, index=slider)


class InferViewer:
    def __init__(
        self,
        infer_fn: Callable[[str | Path], Image.Image],
        image_dir: str | Path,
        glob_pat: str = "*.png",
    ):
        self.image_dir = image_dir
        self.image_files = sorted(Path(image_dir).glob(glob_pat))
        self.infer_fn = infer_fn
        self.current_file = None
        self.current_index = 0

    def view_image_cb(self, index: int):
        self.current_index = index
        self.current_file = self.image_files[index]

        # Call the provided inference function
        image = self.infer_fn(self.current_file)
        print(f"dir={self.image_dir} n_images={len(self.image_files)}")
        print(f"index={index} file={self.current_file.name}")
        display(image)

    def show_widget(self):
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.image_files) - 1,
            description="Image:",
            continuous_update=False,
        )
        widgets.interact(self.view_image_cb, index=slider)


class BBox(TypedDict):
    xyxyn: tuple[float, float, float, float]
    label: str


def bbs_to_df(bboxes: Sequence[BBox], sort=True) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "label": b["label"],
                "x1": b["xyxyn"][0],
                "y1": b["xyxyn"][1],
                "x2": b["xyxyn"][2],
                "y2": b["xyxyn"][3],
            }
            for b in bboxes
        ]
    )
    if sort:
        df = df.sort_values("label").reset_index(drop=True)
    return df


def plot_bb(
    img: Image.Image, bboxes: Sequence[BBox], classes: Sequence[str]
) -> Image.Image:
    """
    Plot bounding boxes
    """
    img = img.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    additional_colors = [colorname for (colorname, _) in ImageColor.colormap.items()]
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ] + additional_colors

    color_map = {classes[i]: colors[i] for i in range(len(classes))}
    for bbox in bboxes:
        color = color_map[bbox["label"]]
        # Convert normalized coordinates to absolute coordinates
        abs_x1 = int(bbox["xyxyn"][0] * width)
        abs_y1 = int(bbox["xyxyn"][1] * height)
        abs_x2 = int(bbox["xyxyn"][2] * width)
        abs_y2 = int(bbox["xyxyn"][3] * height)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # Draw the text
        draw.text((abs_x1 + 8, abs_y1 + 6), bbox["label"], fill=color)

    return img


def gemini_to_bboxes(gemini_bboxes: list[dict[str, Any]]) -> list[BBox]:
    """
    The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.
    [
    {"box_2d": [386, 362, 513, 442], "label": "chicken"},
    {"box_2d": [334, 375, 361, 412], "label": "cow"},
    {"box_2d": [336, 290, 427, 396], "label": "pig"}
    ]
    """

    def _cvt_gemini(gbbox: dict[str, Any]) -> BBox:
        ymin, xmin, ymax, xmax = gbbox["box_2d"]
        xyxyn = (xmin / 1000, ymin / 1000, xmax / 1000, ymax / 1000)
        return BBox(xyxyn=xyxyn, label=gbbox["label"])

    return [_cvt_gemini(_g) for _g in gemini_bboxes]
