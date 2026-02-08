"""
Run model in realtime on Minecraft

uv run mcio inst launch main -w main -W 640 -H 640
uv run fcos_run.py <model_path>

Example:
uv run fcos_run.py ~/data/checkpoints/keep/best.pt
"""

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import mcio_ctrl as mc
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torchvision.transforms.v2 import functional as v2F  # type: ignore

import bb
import tt
import tt_fcos

LOG = logging.getLogger(__name__)


def bb_frame_cb(
    frame: NDArray[np.uint8],
    obs: mc.network.ObservationPacket,
    *,
    trainer: tt_fcos.FCOSTrainer,
) -> NDArray[np.uint8]:
    pt_image = v2F.to_image(frame)
    img = trainer.plot_infer(pt_image)
    return np.array(img)


def load_model(model_path: Path) -> tt_fcos.FCOSTrainer:
    return tt_fcos.FCOSTrainer.load_checkpoint(
        model_path,
        project_dir=model_path.parent,
        score_thresh=0.5,
        nms_thresh=0.4,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FCOS model on Minecraft frames")
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the FCOS model file (e.g., best.pt)",
    )
    args = parser.parse_args()

    tt.logging_init()
    model = load_model(args.model)

    pipeline: mc.mcio_gui.FramePipeline = [
        partial(bb_frame_cb, trainer=model),
        mc.mcio_gui.cursor_frame_cb,
    ]
    gui = mc.mcio_gui.MCioGUI(frame_pipeline=pipeline)
    gui.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
