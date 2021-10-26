"""
This module contains all of the data classes.
Data classes generally consist information that is commonly
used together.
"""
from pathlib import Path

import numpy as np
from wand.image import Image

from image_preprocessing import load_image_as_512_array
from protocols import LazyInitProperty


class ImageData:
    """
    Image data holds data related to a single image.
    File path, Wand Image, Np Array form, Potentially expand to tagging.
    """
    def __init__(self, file_path: Path, img: Image = None, np_array: np.array = None):
        self.file_path: Path = file_path
        self.img: Image = img
        self.np_array: np.array = np_array

    @LazyInitProperty
    def np_array(self) -> np.array:
        """
        Lazy Init version of image as a grayscale np array.
        Lazy Init means this is not calculated on object creation but first access
        to this property.
        :return: 512x512x1 grayscale np array of the image.
        """
        return load_image_as_512_array(self.file_path)
