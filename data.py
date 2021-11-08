"""
This module contains all of the data classes.
Data classes generally consist information that is commonly
used together.
"""
import threading
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

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
        """File where the array is cached."""
        return self.ARRAY_CACHE + self.file_path.name + ".npy"

    def cache_array(self) -> None:
        """Caches the array for easier access in the future."""
        np.save(self.array_cache_file, self.np_array)


class DataSet(ABC):
    """A dataset is an encapsulation of training inputs and expected outputs."""
    def __init__(self):
        self._image_list: list[ImageData] = []
        self._label_list: list[list[str]] = []
        self._list_of_image_arrays: list[np.array] = []
        self._image_array: np.array = np.array([])
        self._debug = False
        self.loaded = False
        self.shuffle = False

    @abstractmethod
    def load_data(self, load_arrays=True) -> None:
        """Abstract method to be implemented by a specific DataSet Should load all data."""

    def _load_arrays(self) -> None:
        """After Image Data loaded this method loads the np_arrays associated with them"""
        for i, item in enumerate(self._image_list):
            self._list_of_image_arrays.append(item.np_array)
        self._image_array = np.array(self._list_of_image_arrays)
        self._list_of_image_arrays = []

    @property
    def images_as_np_array(self) -> np.array:
        """An array of images"""
        return self._image_array

    @images_as_np_array.setter
    def images_as_np_array(self, value: np.array) -> None:
        """Set an array of images"""
        self._image_array = value

    @property
    def labels(self) -> list[list[str]]:
        """List of a list of labels(stings)"""
        return self._label_list

    @labels.setter
    def labels(self, value: list[list[str]]) -> None:
        """Set a List of a list of labels(stings)"""
        self._label_list = value

    @property
    def images(self) -> list[ImageData]:
        """List of ImageData Objects"""
        print(f"{threading.current_thread()}")
        return self._image_list

    @images.setter
    def images(self, value: list[ImageData]) -> None:
        """Set a list of ImageData Objects"""
        self._image_list = value

    def shuffle_array(self) -> None:
        _ = [(i, random.random()) for i in range(len(self._image_list))]
        _.sort(key=lambda x: x[1])
        image_list = []
        array_list = []
        label_list = []
        for i, rand in _:
            image_list.append(self._image_list[i])
            label_list.append(self._label_list[i])
            try:
                array_list.append(self._image_array[i])
            except IndexError:
                pass

        self._image_list = image_list
        self._image_array = np.array(array_list)
        self._label_list = label_list

    def __getitem__(self, item: Union[int, slice]) -> Union[np.array]:
        if isinstance(item, slice):
            return self._slice(item)

    def _slice(self, item: slice):
        new_dataset = self.__class__()
        new_dataset._image_list = self._image_list[item]
        new_dataset._label_list = self._label_list[item]
        new_dataset._image_array = self._image_array[item]
        return new_dataset

    def __len__(self):
        return len(self.images)

    def __add__(self, other):
        self.images = self.images + other.images
        self.labels = self.labels + other.labels
        self._load_arrays()
        return self

    def __iadd__(self, other):
        return self.__add__(other)


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

    def load_data(self, load_arrays=True) -> None:
        """Loads all images and tags"""
        image: ImageData
        self._load_dir(self.root_folder)
        if load_arrays:
            self._load_arrays()
        self.loaded = True


class PersonalPicturesDataSet(DataSet):
    """Loads dataset of tagged personal images"""
    def __init__(self):
        super().__init__()
        self.root_folder = Path("G:/Hashed Pictures")

    def _load_image(self, item: Path):
        """Loads a single image and it's tags"""
        try:
            self._label_list.append(tagging.file_tags(item))
        except KeyError:
            return

        self._image_list.append(ImageData(item))

    def _load_dir(self, folder: Path) -> None:
        """Recursively loads images and directories from a given directory"""
        for item in folder.iterdir():
            if item.is_dir():
                self._load_dir(item)
            elif item.is_file():
                self._load_image(item)

    def load_data(self, load_arrays=True) -> None:
        """Loads all images and tags"""
        image: ImageData
        self._load_dir(self.root_folder)
        if load_arrays:
            self._load_arrays()
        self.loaded = True
