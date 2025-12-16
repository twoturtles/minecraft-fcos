import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import mcio_ctrl as mc
import numpy as np
import ultralytics as ul
from numpy.typing import NDArray
from PIL import Image

import bb
import tt

LOG = logging.getLogger(__name__)


def bb_frame_cb(
    frame: NDArray[np.uint8],
    obs: mc.network.ObservationPacket,
    *,
    model: ul.YOLO,
    dset: bb.Dataset,
) -> NDArray[np.uint8]:
    pred = model(frame, verbose=False)[0]
    bboxes = bb.yr_to_bb(pred)
    img = Image.fromarray(frame, mode="RGB")
    bb.plot_bb_inplace(img, bboxes, dset.categories)
    # print(bboxes)
    return np.array(img)


def load_data() -> tuple[ul.YOLO, bb.Dataset]:
    LOG.info("LOAD-DATA-START")
    data_path = Path.home() / "src/data"
    mc_data_path = data_path / "minecraft/mobs/info.json"
    yolo_path = data_path / "yolo"
    models_path = yolo_path / "models"
    # model_path = Path("/Users/joe/src/data/yolo/runs/detect/keep1/weights/best.pt")
    # model_path = Path("/Users/joe/src/data/yolo/runs/detect/train9/weights/best.pt")
    model_path = Path("/Users/joe/src/data/yolo/runs/detect/keep3/weights/best.pt")

    ul.settings.update(  # type: ignore
        {
            "datasets_dir": str(yolo_path / "datasets"),
            "weights_dir": str(yolo_path / "weights"),
            "runs_dir": str(yolo_path / "runs"),
        }
    )
    dset = bb.Dataset.load(mc_data_path)
    model = ul.YOLO(model_path)
    LOG.info("LOAD-DATA-DONE")
    return model, dset


def main() -> int:
    tt.logging_init()
    model, dset = load_data()

    pipeline: mc.mcio_gui.FramePipeline = [
        partial(bb_frame_cb, model=model, dset=dset),
        mc.mcio_gui.cursor_frame_cb,
    ]
    gui = mc.mcio_gui.MCioGUI(frame_pipeline=pipeline)
    gui.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
