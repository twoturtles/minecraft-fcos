""" utils """

from pathlib import Path
from typing import Callable

from IPython.display import display # type: ignore
import ipywidgets as widgets # type: ignore

from PIL import Image


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
            description='Image:',
            continuous_update=False
        )
        widgets.interact(self.view_image_cb, index=slider)


class InferViewer:
    def __init__(
        self, 
        infer_fn: Callable[[str | Path], Image.Image],
        image_dir: str | Path, 
        glob_pat: str = "*.png"
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
            description='Image:',
            continuous_update=False
        )
        widgets.interact(self.view_image_cb, index=slider)

