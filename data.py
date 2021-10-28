"""
This module contains all of the data classes.
Data classes generally consist information that is commonly
used together.
"""
from pathlib import Path

import numpy as np
from wand.image import Image

from image_preprocessing import load_image_as_256_array
from protocols import LazyInitProperty


class ImageData:
    """
    Image data holds data related to a single image.
    File path, Wand Image, Np Array form, Potentially expand to tagging.
    """
    ARRAY_CACHE = "./Cache/NpArrays/"

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
        if Path(self.array_cache_file).exists():
            return np.load(self.array_cache_file)

        arr = load_image_as_256_array(self.file_path)
        with open(self.array_cache_file, "w+b") as f:
            np.save(f, arr)
        return arr

    @property
    def array_cache_file(self) -> str:
        return self.ARRAY_CACHE + self.file_path.name + ".npy"

    def cache_array(self) -> None:
        np.save(self.array_cache_file, self.np_array)
