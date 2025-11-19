import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from PIL import Image
from tqdm import tqdm

import bb

LOG = logging.getLogger(__name__)


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
) -> list[bb.BBox]:
    """
    Translate gemini format to BBox
    The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.
    [
    {"box_2d": [386, 362, 513, 442], "label": "chicken"},
    {"box_2d": [334, 375, 361, 412], "label": "cow"},
    {"box_2d": [336, 290, 427, 396], "label": "pig"}
    ]
    """

    bboxes: list[bb.BBox] = []
    for gbb in gemini_bboxes:
        ymin, xmin, ymax, xmax = gbb["box_2d"]
        xyxyn = [xmin / 1000, ymin / 1000, xmax / 1000, ymax / 1000]
        cat = gbb["label"]
        if categories is not None and cat not in categories:
            LOG.warning(f"Invalid-Category {cat}")
            continue
        bboxes.append(bb.BBox(xyxyn=xyxyn, category=gbb["label"]))

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
) -> list[bb.BBox]:
    json_result: str = gemini_image_query(image, qcfg)
    gbboxes: list[dict[str, Any]] = json.loads(json_result)
    return gemini_to_bboxes(gbboxes, set(qcfg.categories))


def gemini_detect_gfile(
    image: Image.Image | genai.types.File, qcfg: GeminiQueryConfig, max_tries: int = 3
) -> bb.ImageResult:
    """Pass single bbox query to gemini"""
    from typing import cast

    fname: str
    if isinstance(image, Image.Image):
        fname = image.filename  # type: ignore
        assert isinstance(fname, str)
    else:
        assert image.display_name is not None
        fname = image.display_name
    bboxes: list[bb.BBox] = []
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
    return bb.ImageResult(file=fname, bboxes=bboxes)


def gemini_detect_gfile_multi(
    gfiles: list[genai.types.File], qcfg: GeminiQueryConfig
) -> list[bb.ImageResult]:
    results: list[bb.ImageResult] = []
    for i, gfile in enumerate(gfiles):
        LOG.info(f"Detect index={i} len={len(gfiles)} file={gfile.display_name}")
        assert gfile.display_name is not None
        result = gemini_detect_gfile(gfile, qcfg)
        results.append(result)
    return results
