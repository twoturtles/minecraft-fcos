"""BBox Editing"""

import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeAlias, TypedDict

import ipywidgets as widgets  # type: ignore
import pandas as pd
import torch
import torchvision.transforms.v2.functional as v2F  # type: ignore
from ipydatagrid import DataGrid  # type: ignore
from IPython.display import display
from jupyter_bbox_widget import BBoxWidget  # type: ignore
from PIL import Image
from torchvision import tv_tensors

import bb
import tt

LOG = logging.getLogger(__name__)


"""
Usage in cell:
bedit.DEBUG.clear_output()
bbedit.DEBUG

Usage in widget:
with DEBUG:
    print...
"""
DEBUG = widgets.Output(
    layout={
        # "height": f"{height}px",
        "overflow": "auto",
        "border": "1px solid black",
    }
)


# BBoxEdit format: list of {'x': 377, 'y': 177, 'width': 181, 'height': 201, 'label': 'apple'}
class BBeBB(TypedDict):
    """BBoxEdit BBox"""

    x: int
    y: int
    width: int
    height: int
    label: str


class BBoxEdit:
    def __init__(self, input: str | Path) -> None:
        # Load dataset
        self.file = Path(input)
        self.dset = bb.MCDataset(self.file)

        # Local copy of the current dset item
        self.current_index: int
        self.current_item: bb.MCDatasetItem
        self.current_pil: Image.Image

        self.w = SimpleNamespace()
        self.updating_selection = False  # Prevent cyclic updates
        self.updating_display = False

        # Create all widgets
        initial_index = 0
        bbox = self._create_bbox_panel()
        right_panel = self._create_right_panel(initial_index)
        zoom_section = self._create_zoom_section()

        content_box = widgets.HBox(
            [bbox, right_panel],
            layout={"margin": "0px 0px 20px 0px", "border": "1px solid black"},
        )
        self.w.ui = widgets.VBox([content_box, zoom_section])

        self._set_ui(initial_index)

    def display(self) -> None:
        display(self.w.ui)  # type: ignore[no-untyped-call]

    def save(self, path: str | Path | None = None) -> None:
        if path is None:
            path = self.file
        self.dset.save_annotations(path)

    def _update_display(self) -> None:
        """Update UI from local item copy"""
        if self.updating_display:
            return
        self.updating_display = True

        try:
            image_path = self.dset.image_path(self.current_index)
            self.w.bbox.image = str(image_path)
            self.w.bbox.bboxes = to_bbox_widget(
                self.current_item.target, self.dset.categories
            )
            df = ann_to_df(self.current_item.target, self.dset.categories)
            df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
            self.w.grid.data = df
            self._grid_update_height()
            self._update_zoom()
        finally:
            self.updating_display = False

    def _set_ui(self, index: int) -> None:
        """Load the selected index into the local copy and set the widget"""
        self.current_index = index
        self.current_item = deepcopy(self.dset[index])
        self.current_pil = v2F.to_pil_image(self.current_item.image)
        self._update_display()

    ##
    # BBox edit - left panel

    def _create_bbox_panel(self) -> widgets.Box:
        self.w.bbox = BBoxWidget(
            classes=self.dset.categories,
            colors=tt.Colors().get_strs(),
            hide_buttons=True,
            layout={"width": "60%"},
        )
        self.w.bbox.observe(self._bbox_change_cb, names=["bboxes"])
        self.w.bbox.observe(self._bbox_selection_change_cb, names=["selected_index"])
        return self.w.bbox

    def _bbox_change_cb(self, change: dict[str, Any]) -> None:
        new_bbebb_list: list[BBeBB] = change["new"]
        # update grid with new bboxes (see also _delete_row_cb)
        new_ann = from_bbox_widget(
            new_bbebb_list, self.dset.categories, self.current_pil.size
        )
        self.current_item.target["boxes"] = new_ann["boxes"]
        self.current_item.target["labels"] = new_ann["labels"]
        self._update_display()

    def _bbox_selection_change_cb(self, change: dict[str, Any]) -> None:
        # Update grid with new selection (see also _grid_selection_change)
        if self.updating_selection:
            return
        self.updating_selection = True
        try:
            new_ix = change["new"]
            if new_ix == -1:
                self.w.grid.clear_selection()
            else:
                # In row selection mode, only need to select a single column in the row.
                self.w.grid.select(row1=new_ix, column1=0, clear_mode="all")
        finally:
            self.updating_selection = False

    ##
    # Right panel

    def _create_right_panel(self, initial_index: int) -> widgets.Box:
        control_section = self._create_control_section(initial_index)
        grid_section = self._create_grid_section()
        return widgets.VBox([control_section, grid_section], layout={"width": "40%"})

    ## Control section

    def _create_control_section(self, initial_index: int) -> widgets.Box:
        # Index Slider
        self.w.index_slider = widgets.IntSlider(
            description="Index:",
            value=initial_index,
            min=0,
            max=len(self.dset) - 1,
            step=1,
            continuous_update=False,
        )
        self.w.index_slider.observe(self._on_slider_change, names="value")

        button_box = self._create_buttons()

        control_section = widgets.VBox(
            [self.w.index_slider, button_box],
            layout={
                "border": "1px solid black",
                "padding": "10px",
                "margin": "0px 0px 20px 0px",
            },
        )
        return control_section

    def _on_slider_change(self, change: dict[str, Any]) -> None:
        new_index = change["new"]
        self._set_ui(new_index)

    ## Control Buttons

    def _create_buttons(self) -> widgets.Box:
        back = widgets.Button(
            description="Back", button_style="warning", icon="arrow-left"
        )
        back.on_click(self._on_back)

        submit = widgets.Button(
            description="Submit", button_style="success", icon="check"
        )
        submit.on_click(self._on_submit)

        skip = widgets.Button(
            description="Skip", button_style="warning", icon="arrow-right"
        )
        skip.on_click(self._on_skip)

        save = widgets.Button(
            description="Save",
            button_style="danger",
            icon="save",
            layout={"margin": "2px 2px 2px 20px"},
        )
        save.on_click(self._on_save)

        buttons = [back, submit, skip]
        if self.file is not None:
            buttons.append(save)
        return widgets.HBox(buttons)

    def _on_submit(self, button: widgets.Button) -> None:
        # Update dset from BBoxEdit
        new_ann = from_bbox_widget(
            self.w.bbox.bboxes, self.dset.categories, self.current_pil.size
        )
        self.current_item.target["boxes"] = new_ann["boxes"]
        self.current_item.target["labels"] = new_ann["labels"]
        self.dset.add_annotation(self.current_index, new_ann)
        self._on_skip(button)

    def _on_back(self, button: widgets.Button) -> None:
        slider = self.w.index_slider
        if slider.value > 0:
            slider.value -= 1

    def _on_skip(self, button: widgets.Button) -> None:
        slider = self.w.index_slider
        if slider.value < len(self.dset) - 1:
            slider.value += 1

    def _on_save(self, button: widgets.Button) -> None:
        self.save()

    ## Grid

    def _create_grid_section(self) -> widgets.Box:
        self.w.grid = DataGrid(
            pd.DataFrame(),
            selection_mode="row",
            base_row_size=32,
            base_column_header_size=32,
            auto_fit_columns=True,
            auto_fit_params={"area": "all"},
            layout={"height": "100px"},
        )
        self.w.grid.observe(self._grid_selections_change, "selections")

        delete_row = widgets.Button(
            description="Delete Row", button_style="warning", icon="delete-left"
        )
        delete_row.on_click(self._delete_row_cb)
        return widgets.VBox(
            [self.w.grid, delete_row],
            layout={"border": "1px solid black", "padding": "10px"},
        )

    def _grid_selections_change(self, change: dict[str, Any]) -> None:
        # See also _bbox_selection_change_cb
        if self.updating_selection:
            return
        self.updating_selection = True
        try:
            rows = set([cell["r"] for cell in self.w.grid.selected_cells])
            if len(rows) == 1:
                self.w.bbox.selected_index = rows.pop()
            else:
                # 0 or >1 rows selected. Set bbox to no selection
                self.w.bbox.selected_index = -1
        finally:
            self.updating_selection = False

    def _delete_row_cb(self, button: widgets.Button) -> None:
        # See also _bbox_change_cb
        # list of dicts {'r': 2, 'c': 1}
        size = self.current_pil.size
        rows = set([cell["r"] for cell in self.w.grid.selected_cells])
        if len(rows) > 0:
            new_df = self.w.grid.data.drop(list(rows))
            new_ann = df_to_ann(new_df, self.dset.categories, self.current_pil.size)
            self.current_item.target["boxes"] = new_ann["boxes"]
            self.current_item.target["labels"] = new_ann["labels"]
            self._update_display()

    def _grid_update_height(self) -> None:
        num_rows = len(self.w.grid.data)
        row_height = self.w.grid.base_row_size
        height = (num_rows + 1) * row_height + 10  # header + rows + extra
        self.w.grid.layout = {"height": f"{height}px"}

    ##
    # Zoom panel

    def _create_zoom_section(self) -> widgets.Box:
        self.zoom_level: float = 2.0
        self.zoom_enabled: bool = False

        self.w.zoom_toggle = widgets.Button(
            description="Zoom",
            button_style="info",
            icon="toggle-off",
        )
        self.w.zoom_toggle.on_click(self._on_zoom_toggle)

        zoom_slider = widgets.FloatSlider(
            description="Level",
            value=self.zoom_level,
            min=2.0,
            max=5.0,
            step=0.5,
            continuous_update=False,
        )
        zoom_slider.observe(self._on_zoom_slider_change, names="value")

        zoom_controls = widgets.HBox([self.w.zoom_toggle, zoom_slider])

        self.w.zoom_output = widgets.Output()
        return widgets.VBox(
            [zoom_controls, self.w.zoom_output],
            layout={
                "overflow": "auto",
                "border": "1px solid black",
            },
        )

    def _on_zoom_slider_change(self, change: dict[str, Any]) -> None:
        self.zoom_level = change["new"]
        self._update_zoom()

    def _on_zoom_toggle(self, button: widgets.Button) -> None:
        """Toggle zoom display on/off"""
        self.zoom_enabled = not self.zoom_enabled
        button.icon = "toggle-on" if self.zoom_enabled else "toggle-off"
        self._update_zoom()

    def _update_zoom(self) -> None:
        """Update the zoomed image display."""
        self.w.zoom_output.clear_output()
        if not self.zoom_enabled:
            self.w.zoom_output.layout = {
                "height": "auto",
            }
            return

        # Load and zoom the image
        index = self.w.index_slider.value
        img = self.current_pil
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
            display(zoomed_img)  # type: ignore[no-untyped-call]


##
# Helper functions


def to_bbox_widget(ann: bb.BaseAnnotation, categories: list[str]) -> list[BBeBB]:
    """BoundingBoxes to BBoxEdit format"""
    # BBoxEdit format: list of {'x': 377, 'y': 177, 'width': 181, 'height': 201, 'label': 'apple'}
    boxes_xywh = v2F.convert_bounding_box_format(
        ann["boxes"], new_format=tv_tensors.BoundingBoxFormat.XYWH
    ).tolist()

    bbebb_list: list[BBeBB] = []
    for bbox, label in zip(boxes_xywh, ann["labels"].tolist()):
        category = categories[label]
        bbebb = BBeBB(
            x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3], label=category
        )
        bbebb_list.append(bbebb)
    return bbebb_list


def from_bbox_widget(
    bbebb_list: list[BBeBB], categories: list[str], size: tuple[int, int]
) -> bb.BaseAnnotation:
    """BBoxEdit format to BaseAnnotation.
    size is (width, height) in PIL format.
    """
    boxes_xywh = [[b["x"], b["y"], b["width"], b["height"]] for b in bbebb_list]
    labels = [categories.index(b["label"]) for b in bbebb_list]

    boxes_xyxy = v2F.convert_bounding_box_format(
        torch.tensor(boxes_xywh),
        old_format=tv_tensors.BoundingBoxFormat.XYWH,
        new_format=tv_tensors.BoundingBoxFormat.XYXY,
    )

    w, h = size
    return bb.BaseAnnotation(
        boxes=tv_tensors.BoundingBoxes(
            boxes_xyxy,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        labels=torch.tensor(labels, dtype=torch.int64),
    )


def ann_to_df(ann: bb.BaseAnnotation, categories: list[str]) -> pd.DataFrame:
    """Convert BaseAnnotation to DataFrame with columns: category, x1, y1, x2, y2"""
    # Should already be in XYXY format
    boxes_xyxy = v2F.convert_bounding_box_format(
        ann["boxes"], new_format=tv_tensors.BoundingBoxFormat.XYXY
    ).tolist()
    labels = ann["labels"].tolist()

    rows = [
        {
            "category": categories[label],
            "x1": box[0],
            "y1": box[1],
            "x2": box[2],
            "y2": box[3],
        }
        for box, label in zip(boxes_xyxy, labels)
    ]
    # Specifying columns here for the case when there are no rows (image with no boxes)
    return pd.DataFrame(rows, columns=["category", "x1", "y1", "x2", "y2"])


def df_to_ann(
    df: pd.DataFrame, categories: list[str], size: tuple[int, int]
) -> bb.BaseAnnotation:
    """Convert DataFrame to BaseAnnotation.
    size is (width, height) in PIL format.
    """
    boxes_xyxy = df[["x1", "y1", "x2", "y2"]].values.tolist()
    labels = [categories.index(cat) for cat in df["category"]]

    w, h = size
    return bb.BaseAnnotation(
        boxes=tv_tensors.BoundingBoxes(
            boxes_xyxy,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        labels=torch.tensor(labels, dtype=torch.int64),
    )
