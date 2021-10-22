from Protocols import lazy_init_property
from ImagePreprocessing import *

from pathlib import Path
from wand.image import *

import numpy as np


class ImageData:
    
    def __init__(self, file_path: Path, img: Image = None, np_array: np.array = None):
        self.file_path: Path = file_path
        self.img: Image = img
        self.np_array: np.array = np_array
    
    @lazy_init_property
    def np_array(self) -> np.array:
        return load_image_as_512_array(self.file_path)
