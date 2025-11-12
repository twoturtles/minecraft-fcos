"""utils"""

import copy
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Self, Sequence

import dacite
import ipywidgets as widgets  # type: ignore
import pandas as pd
from google import genai
from IPython.display import display  # type: ignore
from jupyter_bbox_widget import BBoxWidget  # type: ignore
from PIL import Image, ImageDraw
from tqdm import tqdm

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
# Image / BBox


@dataclass(kw_only=True)
class BBox:
    category: str
    xyxyn: list[float]  # len 4

    def to_bbox_widget(self, size: tuple[int, int]) -> dict[str, Any]:
        # {'x': 377, 'y': 177, 'width': 181, 'height': 201, 'label': 'apple'}
        coco = xyxyn_to_coco(self.xyxyn, size)
        return {
            "x": coco[0],
            "y": coco[1],
            "width": coco[2],
            "height": coco[3],
            "label": self.category,
        }

    @classmethod
    def from_bbox_widget(cls, wbbox: dict[str, Any], size: tuple[int, int]) -> Self:
        coco = [
            wbbox["x"],
            wbbox["y"],
            wbbox["width"],
            wbbox["height"],
        ]
        return cls(category=wbbox["label"], xyxyn=coco_to_xyxyn(coco, size))


def sort_xyxy[T: (int, float)](xyxy: list[T]) -> list[T]:
    """Sort corners to ensure x1 < x2 and y1 < y2."""
    x1, y1, x2, y2 = xyxy
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def xyxyn_to_xyxy(xyxyn: list[float], size: tuple[int, int]) -> list[int]:
    """Convert normalized coords to absolute. size is (width, height)"""
    width, height = size
    xyxy = [
        int(xyxyn[0] * width),
        int(xyxyn[1] * height),
        int(xyxyn[2] * width),
        int(xyxyn[3] * height),
    ]
    return sort_xyxy(xyxy)


def xyxy_to_xyxyn(xyxy: list[int], size: tuple[int, int]) -> list[float]:
    width, height = size
    xyxyn = [
        xyxy[0] / width,
        xyxy[1] / height,
        xyxy[2] / width,
        xyxy[3] / height,
    ]
    return sort_xyxy(xyxyn)


def xyxy_to_coco(xyxy: list[int]) -> list[int]:
    # coco is [x, y, width, height] absolute
    return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]


def coco_to_xyxy(coco: list[int]) -> list[int]:
    return [coco[0], coco[1], coco[2] + coco[0], coco[3] + coco[1]]


def xyxyn_to_coco(xyxyn: list[float], size: tuple[int, int]) -> list[int]:
    return xyxy_to_coco(xyxyn_to_xyxy(xyxyn, size))


def coco_to_xyxyn(coco: list[int], size: tuple[int, int]) -> list[float]:
    return xyxy_to_xyxyn(coco_to_xyxy(coco), size)


@dataclass(kw_only=True)
class ImageResult:
    file: str
    bboxes: list[BBox]

    def plot_bb(self, categories: list[str] | None = None) -> Image.Image:
        return plot_bb(Image.open(self.file), self.bboxes, categories)


@dataclass(kw_only=True)
class Dataset:
    categories: list[str]
    images: list[ImageResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, dataset_dict: dict[str, Any]) -> Self:
        return dacite.from_dict(data_class=cls, data=dataset_dict)

    def save(self, path: str | Path):
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy(self) -> Self:
        return copy.deepcopy(self)


# DEBUG_OUT = widgets.Output(
#     layout={
#         "height": "200px",
#         "overflow": "auto",  # Enables scrollbar
#         "border": "1px solid black",
#     }
# )


class BBoxEdit:
    def __init__(self, dset: Dataset) -> None:
        self.dset = dset.copy()

        # Create the slider
        initial_index = 0
        self.w_slider = widgets.IntSlider(
            value=initial_index,
            min=0,
            max=len(self.dset.images) - 1,
            step=1,
            description="Index:",
            continuous_update=False,
        )

        # Create buttons
        self.w_back_button = widgets.Button(
            description="Back", button_style="warning", icon="arrow-left"
        )
        self.w_submit_button = widgets.Button(
            description="Submit", button_style="success", icon="check"
        )
        self.w_skip_button = widgets.Button(
            description="Skip", button_style="warning", icon="arrow-right"
        )

        self.w_bbox = BBoxWidget(
            classes=self.dset.categories,
            colors=Colors().get_strs(),
            hide_buttons=True,
        )
        self._set_bbox(initial_index)

        # Layout the widgets
        self.w_button_box = widgets.HBox(
            [self.w_back_button, self.w_submit_button, self.w_skip_button]
        )
        self.w_ui = widgets.VBox([self.w_slider, self.w_button_box, self.w_bbox])

        # Connect slider to observer
        self.w_slider.observe(self._on_slider_change, names="value")

        # Connect buttons to callbacks
        self.w_back_button.on_click(self._on_back)
        self.w_submit_button.on_click(self._on_submit)
        self.w_skip_button.on_click(self._on_skip)

    def display(self) -> None:
        display(self.w_ui)

    # Callbacks

    def _set_bbox(self, index: int) -> None:
        """Update bbox widget to current slider index."""
        image_result = self.dset.images[index]
        self.w_bbox.image = str(image_result.file)
        size = Image.open(image_result.file).size  # XXX
        self.w_bbox.bboxes = [bbox.to_bbox_widget(size) for bbox in image_result.bboxes]

    def _on_slider_change(self, change: dict[str, Any]) -> None:
        new_index = change["new"]
        self._set_bbox(new_index)

    def _on_submit(self, button: widgets.Button) -> None:
        index: int = self.w_slider.value
        image_result = self.dset.images[index]
        size = Image.open(image_result.file).size  # XXX
        display(f"DO SUBMIT {index}")
        display(self.dset.images[index].bboxes)
        display(self.w_bbox.bboxes)
        new = [BBox.from_bbox_widget(bb, size) for bb in self.w_bbox.bboxes]
        display(new)
        self._on_skip(button)

    def _on_back(self, button: widgets.Button) -> None:
        slider = self.w_slider
        if slider.value > 0:
            slider.value -= 1

    def _on_skip(self, button: widgets.Button) -> None:
        slider = self.w_slider
        if slider.value < len(self.dset.images) - 1:
            slider.value += 1


##
# Viewers


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

    def view_image_cb(self, index: int):
        # Call the provided inference function
        result = self.infer_fn(self.infer_list[index])
        print(f"index={index} file={result.file}")
        display(result.plot_bb(categories=self.categories))

    def show_widget(self):
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.infer_list) - 1,
            description="Image:",
            continuous_update=False,
        )
        widgets.interact(self.view_image_cb, index=slider)


def bbs_to_df(bboxes: Sequence[BBox], sort: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "category": b.category,
                "x1": b.xyxyn[0],
                "y1": b.xyxyn[1],
                "x2": b.xyxyn[2],
                "y2": b.xyxyn[3],
            }
            for b in bboxes
        ]
    )
    if sort:
        df = df.sort_values("category").reset_index(drop=True)
    return df


def plot_bb(
    img: Image.Image, bboxes: Sequence[BBox], categories: Sequence[str] | None
) -> Image.Image:
    """
    Plot bounding boxes
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    if categories is None:
        # Use labels to create categories set.
        categories = sorted(list(set([bbox.category for bbox in bboxes])))

    colors = Colors().get_rgb()
    color_map = {categories[i]: colors[i] for i in range(len(categories))}
    for bbox in bboxes:
        color = color_map[bbox.category]
        xyxy = xyxyn_to_xyxy(bbox.xyxyn, img.size)
        draw.rectangle(((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])), outline=color, width=2)
        draw.text((xyxy[0] + 4, xyxy[1] + 2), bbox.category, fill=color, font_size=16)

    return img


##
# Gemini


@dataclass(kw_only=True)
class GeminiQueryConfig:
    prompt: str
    categories: list[str]
    model: str = "gemini-2.5-flash"
    temperature: float | None = 0.0
    seed: int | None = 325


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
        xyxyn = [xmin / 1000, ymin / 1000, xmax / 1000, ymax / 1000]
        return BBox(xyxyn=xyxyn, category=gbbox["label"])

    return [_cvt_gemini(_g) for _g in gemini_bboxes]


class GeminiFileAPI:

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
        for file in tqdm(files):
            self.upload_file(file)

    def rm(self, name: str) -> None:
        self.client.files.delete(name=name)

    def clear(self) -> None:
        self.sync()
        for gfile in self.gfiles:
            assert gfile.name is not None
            self.rm(name=gfile.name)
        self.gfiles = []


"""
TODO:
https://ai.google.dev/gemini-api/docs/batch-api
The Gemini Batch API is designed to process large volumes of requests
asynchronously at 50% of the standard cost. The target turnaround time is 24
hours, but in majority of cases, it is much quicker.
"""


def gemini_gen_bboxes(
    image: Image.Image | genai.types.File, qcfg: GeminiQueryConfig
) -> list[BBox]:
    client = genai.Client()
    config = genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        temperature=qcfg.temperature,
        seed=qcfg.seed,
    )
    response = client.models.generate_content(
        model=qcfg.model, contents=[image, qcfg.prompt], config=config
    )
    assert response.text is not None
    LOG.info(f"RESPONSE len={len(response.text)} {response.text}")
    bounding_boxes: list[dict[str, Any]] = json.loads(response.text)
    return gemini_to_bboxes(bounding_boxes)


def gemini_detect_multi(
    gfiles: list[genai.types.File], qcfg: GeminiQueryConfig
) -> list[ImageResult]:
    results: list[ImageResult] = []
    for i, gfile in enumerate(tqdm(gfiles)):
        LOG.info(f"Detect {i+1}/{len(gfiles)} file={gfile.display_name}")
        assert gfile.display_name is not None
        result = gemini_detect_gfile(gfile, qcfg)
        results.append(result)
    return results


def gemini_detect_gfile(
    gfile: genai.types.File, qcfg: GeminiQueryConfig
) -> ImageResult:
    assert gfile.display_name is not None
    bbs = gemini_gen_bboxes(gfile, qcfg)
    return ImageResult(file=gfile.display_name, bboxes=bbs)


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
        """Get a color scheme as a tuple of RGB tuples."""
        # Convert each hex color to RGB tuple
        rgb_colors: list[tuple[int, int, int]] = []
        for h in self.hex_split(scheme):
            rgb_colors.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
        return rgb_colors

    def get_strs(self, scheme: str = "category20") -> list[str]:
        hex_list = self.hex_split(scheme)
        return [f"#{color}" for color in hex_list]

    def hex_split(self, scheme: str):
        # Split into 6-character chunks
        hex_string = self.schemes[scheme]
        return [hex_string[i : i + 6] for i in range(0, len(hex_string), 6)]
