"""utils"""

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Self, Sequence

import dacite
import ipywidgets as widgets  # type: ignore
import pandas as pd
from google import genai
from ipydatagrid import DataGrid  # type: ignore
from IPython.display import display
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
# BBox


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
        xyxyn = coco_to_xyxyn(coco, size)
        # BBoxWidget can return values out of range if you overlap the edge with a box
        xyxyn = [clamp(bb, 0.0, 1.0) for bb in xyxyn]
        return cls(category=wbbox["label"], xyxyn=xyxyn)


@dataclass(kw_only=True)
class ImageResult:
    file: str
    bboxes: list[BBox]

    def plot_bb(self, categories: list[str] | None = None) -> Image.Image:
        return plot_bb(Image.open(self.file), self.bboxes, categories)

    def to_df(self, sort: bool = True) -> pd.DataFrame:
        df = pd.DataFrame(
            [[bbox.category, *bbox.xyxyn] for bbox in self.bboxes],
            columns=["category", "x1", "y1", "x2", "y2"],
        )
        if sort:
            df = df.sort_values("category").reset_index(drop=True)
        return df


@dataclass(kw_only=True)
class Dataset:
    categories: list[str]
    images: list[ImageResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, dataset_dict: dict[str, Any]) -> Self:
        return dacite.from_dict(data_class=cls, data=dataset_dict)

    def save(self, path: str | Path) -> None:
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

    colors = Colors().get_rgb()
    color_map = {categories[i]: colors[i] for i in range(len(categories))}
    for bbox in bboxes:
        color = color_map[bbox.category]
        xyxy = xyxyn_to_xyxy(bbox.xyxyn, img.size)
        draw.rectangle(((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])), outline=color, width=2)
        draw.text((xyxy[0] + 4, xyxy[1] + 2), bbox.category, fill=color, font_size=16)

    return img


class BBoxEdit:
    def __init__(self, input: str | Path | Dataset) -> None:
        # Load dataset
        if isinstance(input, Dataset):
            self.file = None
            self.dset = input.copy()
        else:
            self.file = Path(input)
            self.dset = Dataset.load(self.file)

        initial_index = 0
        self.w = SimpleNamespace()
        self.zoom_level: float = 1.0

        # Create all widgets
        self._create_index_slider(initial_index)
        self._create_buttons()
        self._create_grid()
        self._create_zoom()
        self._create_bbox(initial_index)
        self._create_debug()
        self._build_layout()

    def _create_index_slider(self, initial_index: int) -> None:
        self.w.slider = widgets.IntSlider(
            description="Index:",
            value=initial_index,
            min=0,
            max=len(self.dset.images) - 1,
            step=1,
            continuous_update=False,
        )
        self.w.slider.observe(self._on_slider_change, names="value")

    def _create_buttons(self) -> None:
        self.w.back = widgets.Button(
            description="Back", button_style="warning", icon="arrow-left"
        )
        self.w.back.on_click(self._on_back)

        self.w.submit = widgets.Button(
            description="Submit", button_style="success", icon="check"
        )
        self.w.submit.on_click(self._on_submit)

        self.w.skip = widgets.Button(
            description="Skip", button_style="warning", icon="arrow-right"
        )
        self.w.skip.on_click(self._on_skip)

        self.w.save = widgets.Button(
            description="Save",
            button_style="danger",
            icon="save",
            layout=widgets.Layout(margin="2px 2px 2px 20px"),
        )
        self.w.save.on_click(self._on_save)

    def _create_grid(self) -> None:
        self.w.grid = DataGrid(
            pd.DataFrame(),
            editable=True,
            selection_mode="row",
            base_row_size=32,
            base_column_header_size=32,
            auto_fit_columns=True,
            auto_fit_params={"area": "all"},
            layout={"height": "100px"},
        )
        self.w.grid.on_cell_change(self._grid_change_cb)
        self.w.delete_row = widgets.Button(
            description="Delete Row", button_style="warning", icon="delete-left"
        )

    def _create_zoom(self) -> None:
        self.w.zoom_slider = widgets.FloatSlider(
            description="Zoom",
            value=1.0,
            min=1.0,
            max=5.0,
            step=0.5,
            continuous_update=False,
        )
        self.w.zoom_slider.observe(self._on_zoom_slider_change, names="value")
        self.w.zoom_output = widgets.Output(
            layout={"height": "50px", "border": "1px solid black"}
        )

    def _create_bbox(self, initial_index: int) -> None:
        self.w.bbox = BBoxWidget(
            classes=self.dset.categories,
            colors=Colors().get_strs(),
            hide_buttons=True,
        )
        self.w.bbox.layout = widgets.Layout(width="60%", border="1px solid black")
        self._set_bbox(initial_index)

    def _create_debug(self) -> None:
        self.w.debug = widgets.Output(
            layout={
                "height": "200px",
                "overflow": "auto",
                "border": "1px solid black",
            }
        )

    def _build_layout(self) -> None:
        buttons = [self.w.back, self.w.submit, self.w.skip]
        if self.file is not None:
            buttons.append(self.w.save)

        self.w.button_box = widgets.HBox(buttons, layout={"margin": "0px 0px 50px 0px"})

        self.w.top_right_panel = widgets.VBox(
            [self.w.slider, self.w.button_box, self.w.grid, self.w.delete_row],
            layout={"height": "90%"},
        )

        self.w.right_panel = widgets.VBox(
            [self.w.top_right_panel, self.w.zoom_slider],
            layout=widgets.Layout(width="40%"),
        )

        self.w.content_box = widgets.HBox([self.w.bbox, self.w.right_panel])
        self.w.ui = widgets.VBox([self.w.content_box])

    def display(self) -> None:
        display(self.w.ui)  # type: ignore
        display(self.w.zoom_output)
        # display(self.w.debug)

    def save(self, path: str | Path | None = None) -> None:
        if path is None:
            path = self.file
        assert path is not None
        self.dset.save(path)

    # Callbacks

    def _set_bbox(self, index: int) -> None:
        """Update bbox widget to current slider index."""
        image_result = self.dset.images[index]
        self.w.bbox.image = str(image_result.file)
        size = Image.open(image_result.file).size  # XXX
        self.w.bbox.bboxes = [bbox.to_bbox_widget(size) for bbox in image_result.bboxes]
        self.w.grid.data = image_result.to_df()
        self._update_zoom()

    def _on_slider_change(self, change: dict[str, Any]) -> None:
        new_index = change["new"]
        self._set_bbox(new_index)

    def _on_submit(self, button: widgets.Button) -> None:
        index: int = self.w.slider.value
        image_result = self.dset.images[index]
        size = Image.open(image_result.file).size  # XXX
        new = [BBox.from_bbox_widget(bb, size) for bb in self.w.bbox.bboxes]
        self.dset.images[index].bboxes = new
        self._on_skip(button)

    def _on_back(self, button: widgets.Button) -> None:
        slider = self.w.slider
        if slider.value > 0:
            slider.value -= 1

    def _on_skip(self, button: widgets.Button) -> None:
        slider = self.w.slider
        if slider.value < len(self.dset.images) - 1:
            slider.value += 1

    def _on_save(self, button: widgets.Button) -> None:
        self.save()

    def _grid_change_cb(self, cell: dict[str, Any]) -> None:
        print("Cell change")
        print(cell)

    def _on_zoom_slider_change(self, change: dict[str, Any]) -> None:
        self.zoom_level = change["new"]
        self._update_zoom()

    def _update_zoom(self) -> None:
        """Update the zoomed image display."""
        # with self.w.debug:
        self.w.zoom_output.clear_output()
        if self.zoom_level <= 1.0:
            self.w.zoom_output.layout = {
                "height": "50px",
                "border": "1px solid black",
            }
            return

        # Load and zoom the image
        index = self.w.slider.value
        image_result = self.dset.images[index]
        img = Image.open(image_result.file)
        zoomed_img = img.resize(
            (int(img.width * self.zoom_level), int(img.height * self.zoom_level)),
            Image.Resampling.LANCZOS,
        )
        self.w.zoom_output.layout = {
            "height": f"{zoomed_img.height}px",
            "width": f"{zoomed_img.width}px",
            "border": "1px solid black",
        }

        with self.w.zoom_output:
            display(zoomed_img)


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
        display(result.plot_bb(categories=self.categories))

    def show_widget(self) -> None:
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


##
# Gemini


@dataclass(kw_only=True)
class GeminiQueryConfig:
    client: genai.Client = field(default_factory=genai.Client)
    prompt: str
    categories: list[str]
    model: str = "gemini-2.5-flash"
    temperature: float | None = 0.0
    seed: int | None = 325


class GeminiFileAPI:

    def __init__(self, client: genai.Client | None = None, sync: bool = True) -> None:
        if client is None:
            client = genai.Client()
        self.client = client
        self.gfiles: list[genai.types.File] = []
        if sync:
            self.sync()

    def sync(self) -> None:
        self.gfiles = list(self.client.files.list())

    def upload_file(self, file: str | Path) -> genai.types.File:
        path = Path(file).absolute()
        gfile = self.client.files.upload(
            file=path,
            config=genai.types.UploadFileConfig(display_name=str(path)),
        )
        return gfile

    def upload_dir(
        self,
        dir_path: str | Path,
        glob_pat: str = "*.png",
        slice_obj: slice = slice(None),
    ) -> None:
        dir_path = Path(dir_path).absolute()
        files = sorted(dir_path.glob(glob_pat))
        files = files[slice_obj]
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


def gemini_to_bboxes(
    gemini_bboxes: list[dict[str, Any]], categories: set[str] | None
) -> list[BBox]:
    """
    Translate gemini format to BBox
    The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.
    [
    {"box_2d": [386, 362, 513, 442], "label": "chicken"},
    {"box_2d": [334, 375, 361, 412], "label": "cow"},
    {"box_2d": [336, 290, 427, 396], "label": "pig"}
    ]
    """

    bboxes: list[BBox] = []
    for gbb in gemini_bboxes:
        ymin, xmin, ymax, xmax = gbb["box_2d"]
        xyxyn = [xmin / 1000, ymin / 1000, xmax / 1000, ymax / 1000]
        cat = gbb["label"]
        if categories is not None and cat not in categories:
            LOG.warning(f"Invalid-Category {cat}")
            continue
        bboxes.append(BBox(xyxyn=xyxyn, category=gbb["label"]))

    return bboxes


def gemini_image_query(
    image: Image.Image | genai.types.File, qcfg: GeminiQueryConfig
) -> str:
    """Pass image to gemini and return the text result"""
    config = genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        temperature=qcfg.temperature,
        seed=qcfg.seed,
    )
    response = qcfg.client.models.generate_content(
        model=qcfg.model, contents=[image, qcfg.prompt], config=config
    )
    assert response.text is not None
    LOG.info(f"Response len={len(response.text)} {response.text}")
    return response.text


def gemini_detect_single(
    image: Image.Image | genai.types.File, qcfg: GeminiQueryConfig
) -> list[BBox]:
    json_result: str = gemini_image_query(image, qcfg)
    gbboxes: list[dict[str, Any]] = json.loads(json_result)
    return gemini_to_bboxes(gbboxes, set(qcfg.categories))


def gemini_detect_gfile(
    image: Image.Image | genai.types.File, qcfg: GeminiQueryConfig, max_tries: int = 3
) -> ImageResult:
    """Pass single bbox query to gemini"""
    from typing import cast

    fname: str
    if isinstance(image, Image.Image):
        fname = image.filename  # type: ignore
        assert isinstance(fname, str)
    else:
        assert image.display_name is not None
        fname = image.display_name
    bboxes: list[BBox] = []
    for n in range(max_tries):
        LOG.info(f"Query-Attempt-{n+1}")
        json_result: str = gemini_image_query(image, qcfg)
        try:
            gbboxes: list[dict[str, Any]] = json.loads(json_result)
        except json.JSONDecodeError as e:
            LOG.warning(repr(e))
            continue
        bboxes = gemini_to_bboxes(gbboxes, set(qcfg.categories))
        break
    return ImageResult(file=fname, bboxes=bboxes)


def gemini_detect_gfile_multi(
    gfiles: list[genai.types.File], qcfg: GeminiQueryConfig
) -> list[ImageResult]:
    results: list[ImageResult] = []
    for i, gfile in enumerate(gfiles):
        LOG.info(f"Detect index={i} len={len(gfiles)} file={gfile.display_name}")
        assert gfile.display_name is not None
        result = gemini_detect_gfile(gfile, qcfg)
        results.append(result)
    return results


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

    def hex_split(self, scheme: str) -> list[str]:
        # Split into 6-character chunks
        hex_string = self.schemes[scheme]
        return [hex_string[i : i + 6] for i in range(0, len(hex_string), 6)]
