"""utils"""

import logging
from pathlib import Path
from typing import Any, Callable, Self, Sequence, TypedDict

import ipywidgets as widgets  # type: ignore
import pandas as pd
from google import genai
from IPython.display import display  # type: ignore
from PIL import Image, ImageColor, ImageDraw

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


class InferViewer[T]:
    def __init__(
        self,
        infer_fn: Callable[[T], tuple[Image.Image, str]],
        infer_list: list[T],
    ):
        # self.image_dir = image_dir
        # self.image_files = sorted(Path(image_dir).glob(glob_pat))
        self.infer_fn = infer_fn
        self.infer_list = infer_list
        # self.current_file = None
        # self.current_index = 0

    def view_image_cb(self, index: int):
        # Call the provided inference function
        image, description = self.infer_fn(self.infer_list[index])
        print(f"index={index} desc={description}")
        display(image)

    def show_widget(self):
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.infer_list) - 1,
            description="Image:",
            continuous_update=False,
        )
        widgets.interact(self.view_image_cb, index=slider)


class BBox(TypedDict):
    xyxyn: tuple[float, float, float, float]
    label: str


def bbs_to_df(bboxes: Sequence[BBox], sort: bool = True) -> pd.DataFrame:
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
        draw.text((abs_x1 + 8, abs_y1 + 6), bbox["label"], fill=color, font_size=16)

    return img


##
# Gemini


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


class GeminiFile:

    def __init__(self, client: genai.Client | None = None, sync: bool = True) -> None:
        if client is None:
            client = genai.Client()
        self.client = client
        self.gfiles: list[genai.types.File] = []
        if sync:
            self.sync()

    def sync(self):
        self.gfiles = list(self.client.files.list())

    def upload_file(self, file: str | Path) -> genai.types.File:
        path = Path(file).absolute()
        gfile = self.client.files.upload(
            file=path,
            config=genai.types.UploadFileConfig(display_name=str(path)),
        )
        return gfile

    def upload_dir(self, dir_path: str | Path, glob_pat: str = "*.png"):
        dir_path = Path(dir_path).absolute()
        files = sorted(dir_path.glob(glob_pat))
        for file in files:
            self.upload_file(file)

    def rm(self, name: str) -> None:
        self.client.files.delete(name=name)

    def clear(self) -> None:
        self.sync()
        for gfile in self.gfiles:
            assert gfile.name is not None
            self.rm(name=gfile.name)
        self.gfiles = []
