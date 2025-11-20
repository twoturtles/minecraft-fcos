"""BBox Editing"""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeAlias

import ipywidgets as widgets  # type: ignore
import pandas as pd
from ipydatagrid import DataGrid  # type: ignore
from IPython.display import display
from jupyter_bbox_widget import BBoxWidget  # type: ignore
from PIL import Image

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
BBeBB = dict[str, Any]  # BBoxEdit BBox


class BBoxEdit:
    def __init__(self, input: str | Path | bb.Dataset) -> None:
        # Load dataset
        if isinstance(input, bb.Dataset):
            self.file = None
            self.dset = input.copy()
        else:
            self.file = Path(input)
            self.dset = bb.Dataset.load(self.file)

        initial_index = 0
        self.w = SimpleNamespace()
        self.current_image = Image.Image()
        self.updating_selection = False  # Prevent cyclic updates

        # Create all widgets
        bbox = self._create_bbox_panel()
        right_panel = self._create_right_panel(initial_index)
        zoom_section = self._create_zoom_section()

        content_box = widgets.HBox(
            [bbox, right_panel],
            layout={"margin": "0px 0px 20px 0px", "border": "1px solid black"},
        )
        self.w.ui = widgets.VBox([content_box, zoom_section])

        self._set_ui_from_index(initial_index)

    def display(self) -> None:
        display(self.w.ui)  # type: ignore[no-untyped-call]

    def save(self, path: str | Path | None = None) -> None:
        if path is None:
            path = self.file
        assert path is not None
        self.dset.save(path)

    def _set_ui_from_ir(self, image_result: bb.ImageResult) -> None:
        """Update UI from passed ImageResult"""
        self.current_image = Image.open(image_result.file)
        size = self.current_image.size
        self.w.bbox.image = str(image_result.file)
        self.w.bbox.bboxes = to_bbox_widget(image_result.bboxes, size)
        self.w.grid.data = image_result.to_df()
        self._update_zoom()

    def _set_ui_from_index(self, index: int) -> None:
        """Update bbox widget to selected index."""
        image_result = self.dset.images[index]
        self._set_ui_from_ir(image_result)

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
        new_bb_list = from_bbox_widget(new_bbebb_list, self.current_image.size)
        new_ir = bb.ImageResult(file=self.w.bbox.image, bboxes=new_bb_list)
        self._set_ui_from_ir(new_ir)

    def _bbox_selection_change_cb(self, change: dict[str, Any]) -> None:
        # Update grid with new selection (see also _grid_selection_change)
        if self.updating_selection:
            return
        self.updating_selection = True
        try:
            new_ix = change["new"]
            with DEBUG:
                print(f"BBOX CALLBACK new_ix={new_ix}")
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
            max=len(self.dset.images) - 1,
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
        self._set_ui_from_index(new_index)

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
        index: int = self.w.index_slider.value
        size = self.current_image.size
        # Update ImageResult from BBoxEdit
        new_bbs = from_bbox_widget(self.w.bbox.bboxes, size)
        self.dset.images[index].bboxes = new_bbs
        self._on_skip(button)

    def _on_back(self, button: widgets.Button) -> None:
        slider = self.w.index_slider
        if slider.value > 0:
            slider.value -= 1

    def _on_skip(self, button: widgets.Button) -> None:
        slider = self.w.index_slider
        if slider.value < len(self.dset.images) - 1:
            slider.value += 1

    def _on_save(self, button: widgets.Button) -> None:
        self.save()

    ## Grid

    def _create_grid_section(self) -> widgets.Box:
        # XXX Turn off edit?
        self.w.grid = DataGrid(
            pd.DataFrame(),
            selection_mode="row",
            base_row_size=32,
            base_column_header_size=32,
            auto_fit_columns=True,
            auto_fit_params={"area": "all"},
            layout={"height": "150px"},
        )
        self.w.grid.observe(self._grid_selection_change, "selections")

        delete_row = widgets.Button(
            description="Delete Row", button_style="warning", icon="delete-left"
        )
        delete_row.on_click(self._delete_row_cb)
        return widgets.VBox(
            [self.w.grid, delete_row],
            layout={"border": "1px solid black", "padding": "10px"},
        )

    def _grid_selection_change(self, change: dict[str, Any]) -> None:
        # See also _bbox_selection_change_cb
        if self.updating_selection:
            return
        self.updating_selection = True
        try:
            rows = set([cell["r"] for cell in self.w.grid.selected_cells])
            with DEBUG:
                print(f"GRID CALLBACK rows={rows}")
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
        rows = set([cell["r"] for cell in self.w.grid.selected_cells])
        if len(rows) > 0:
            new_df = self.w.grid.data.drop(list(rows))
            new_ir = bb.ImageResult.from_df(new_df, self.w.bbox.image)
            self._set_ui_from_ir(new_ir)

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
        image_result = self.dset.images[index]
        img = self.current_image
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


def to_bbox_widget(bboxes: list[bb.BBox], size: tuple[int, int]) -> list[BBeBB]:
    """list[BBox] to BBoxEdit format"""
    # BBoxEdit format: list of {'x': 377, 'y': 177, 'width': 181, 'height': 201, 'label': 'apple'}
    bbebb_list: list[BBeBB] = []
    for bbox in bboxes:
        coco = bb.xyxyn_to_coco(bbox.xyxyn, size)
        bbebb = {
            "x": coco[0],
            "y": coco[1],
            "width": coco[2],
            "height": coco[3],
            "label": bbox.category,
        }
        bbebb_list.append(bbebb)
    return bbebb_list


def from_bbox_widget(bbebb_list: list[BBeBB], size: tuple[int, int]) -> list[bb.BBox]:
    bb_list: list[bb.BBox] = []
    for bbebb in bbebb_list:
        coco = [
            bbebb["x"],
            bbebb["y"],
            bbebb["width"],
            bbebb["height"],
        ]
        xyxyn = bb.coco_to_xyxyn(coco, size)
        # BBoxEdit can return values out of range if you overlap the edge with a box
        xyxyn = [bb.clamp(val, 0.0, 1.0) for val in xyxyn]
        bb_list.append(bb.BBox(category=bbebb["label"], xyxyn=xyxyn))
    return bb_list
