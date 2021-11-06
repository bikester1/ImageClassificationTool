"""
This module contains all of the data classes.
Data classes generally consist information that is commonly
used together.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from wand.image import Image

from tagging import tagging
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


class DataSet(ABC):

    def __init__(self):
        self._image_list: list[ImageData] = []
        self._label_list: list[list[str]] = []
        self._list_of_image_arrays: list[np.array] = []
        self._image_array: np.array = np.array([])
        self._debug = False
        self.loaded = False

    @abstractmethod
    def load_data(self) -> None:
        pass

    @property
    def images_as_np_array(self) -> np.array:
        if not self.loaded:
            self.load_data()
        return self._image_array

    @images_as_np_array.setter
    def images_as_np_array(self, value: np.array) -> None:
        self._image_array = value

    @property
    def labels(self) -> list[list[str]]:
        if not self.loaded:
            self.load_data()
        return self._label_list

    @labels.setter
    def labels(self, value: list[str]) -> None:
        self._label_list = value

    @property
    def images(self) -> np.array:
        if not self.loaded:
            self.load_data()
        return self._image_list

    @images.setter
    def images(self, value: list[ImageData]) -> None:
        self._image_list = value

    def __len__(self):
        return len(self.images)

    def __add__(self, other):
        self.images = self.images + other.images
        self.labels = self.labels + other.labels

    def __iadd__(self, other):
        self.__add__(other)


class DogDataSet(DataSet):
    """Loads the Stanford Dogs Dataset and tags"""

    def __init__(self):
        super().__init__()
        self.root_folder = Path("Datasets/Dogs/Images")

    def _load_image(self, item: Path):
        """Loads a single image and it's tags"""
        self._image_list.append(ImageData(item))
        self._label_list.append([item.parts[-2][10:].lower(), "dog"])

    def _load_dir(self, folder: Path) -> None:
        """Recursively loads images and directories from a given directory"""
        for item in folder.iterdir():
            if item.is_dir():
                self._load_dir(item)
            elif item.is_file():
                self._load_image(item)

    def _load_arrays(self) -> None:
        """After Image Data loaded this method loads the np_arrays associated with them"""
        for i, item in enumerate(self._image_list):
            self._list_of_image_arrays.append(item.np_array)
        self._image_array = np.array(self._list_of_image_arrays)
        self._list_of_image_arrays = None

    def load_data(self) -> None:
        """Loads all images and tags"""
        image: ImageData
        self._load_dir(self.root_folder)
        if self._debug:
            print(f"Loading as array.")
        self._load_arrays()
        self.loaded = True


class PersonalPicturesDataSet(DataSet):
    """Loads dataset of tagged personal images"""

    def __init__(self):
        super().__init__()
        self.root_folder = Path("G:/Hashed Pictures")

    def _load_image(self, item: Path):
        """Loads a single image and it's tags"""
        self._image_list.append(ImageData(item))
        self._label_list.append([tagging.file_tags(item)])

    def _load_dir(self, folder: Path) -> None:
        """Recursively loads images and directories from a given directory"""
        for item in folder.iterdir():
            if item.is_dir():
                self._load_dir(item)
            elif item.is_file():
                self._load_image(item)

    def _load_arrays(self) -> None:
        """After Image Data loaded this method loads the np_arrays associated with them"""
        for i, item in enumerate(self._image_list):
            self._list_of_image_arrays.append(item.np_array)
        self._image_array = np.array(self._list_of_image_arrays)
        self._list_of_image_arrays = None

    def load_data(self) -> None:
        """Loads all images and tags"""
        image: ImageData
        self._load_dir(self.root_folder)
        if self._debug:
            print(f"Loading as array.")
        self._load_arrays()
        self.loaded = True
