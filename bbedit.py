import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ipywidgets as widgets  # type: ignore
import pandas as pd
from ipydatagrid import DataGrid  # type: ignore
from IPython.display import display
from jupyter_bbox_widget import BBoxWidget  # type: ignore
from PIL import Image

import bb
import tt

LOG = logging.getLogger(__name__)


def debug(height: int = 250) -> widgets.Output:
    """
    Usage:
    # Cell 1
    debug = bbedit.debug()
    # Cell 2
    with debug:
        ...
    """
    out = widgets.Output(
        layout={
            "height": f"{height}px",
            "overflow": "auto",
            "border": "1px solid black",
        }
    )
    display(out)  # type: ignore[no-untyped-call]
    return out


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

        # Create all widgets
        self.w.bbox = self._create_bbox_panel()
        right_panel = self._create_right_panel(initial_index)
        zoom_section = self._create_zoom_section()

        content_box = widgets.HBox(
            [self.w.bbox, right_panel],
            layout={"margin": "0px 0px 20px 0px", "border": "1px solid black"},
        )
        self.w.ui = widgets.VBox([content_box, zoom_section])

        self._set_bbox(initial_index)

    ##
    # BBox Edit - left panel
    def _create_bbox_panel(self) -> widgets.Box:
        bbox = BBoxWidget(
            classes=self.dset.categories,
            colors=tt.Colors().get_strs(),
            hide_buttons=True,
            layout={"width": "60%"},
        )
        return bbox

    ##
    # Right panel

    def _create_right_panel(self, initial_index: int) -> widgets.Box:
        control_section = self._create_control_section(initial_index)
        grid_section = self._create_grid_section()
        return widgets.VBox([control_section, grid_section], layout={"width": "40%"})

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

    def _create_grid_section(self) -> widgets.Box:
        self.w.grid = DataGrid(
            pd.DataFrame(),
            editable=True,
            selection_mode="row",
            base_row_size=32,
            base_column_header_size=32,
            auto_fit_columns=True,
            auto_fit_params={"area": "all"},
            layout={"height": "150px"},
        )
        self.w.grid.on_cell_change(self._grid_change_cb)
        self.w.delete_row = widgets.Button(
            description="Delete Row", button_style="warning", icon="delete-left"
        )
        return widgets.VBox(
            [self.w.grid, self.w.delete_row],
            layout={"border": "1px solid black", "padding": "10px"},
        )

    ##
    # Zoom panel

    def _create_zoom_section(self) -> widgets.Box:
        self.zoom_level: float = 1.0
        zoom_slider = widgets.FloatSlider(
            description="Zoom",
            value=1.0,
            min=1.0,
            max=5.0,
            step=0.5,
            continuous_update=False,
        )
        zoom_slider.observe(self._on_zoom_slider_change, names="value")
        self.w.zoom_output = widgets.Output()
        return widgets.VBox(
            [zoom_slider, self.w.zoom_output],
            layout={
                "overflow": "auto",
                "border": "1px solid black",
            },
        )

    def display(self) -> None:
        display(self.w.ui)  # type: ignore[no-untyped-call]

        # display(self.w.debug)

    def save(self, path: str | Path | None = None) -> None:
        if path is None:
            path = self.file
        assert path is not None
        self.dset.save(path)

    ##
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
        index: int = self.w.index_slider.value
        image_result = self.dset.images[index]
        size = Image.open(image_result.file).size  # XXX
        new = [bb.BBox.from_bbox_widget(bb, size) for bb in self.w.bbox.bboxes]
        self.dset.images[index].bboxes = new
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
                "height": "auto",
            }
            return

        # Load and zoom the image
        index = self.w.index_slider.value
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
            display(zoomed_img)  # type: ignore[no-untyped-call]
