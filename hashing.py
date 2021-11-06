"""This module contains code relating to hashing of images and videos."""
import os
import re
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Union
from tempfile import TemporaryFile

import PIL.Image
from PIL.ImageStat import Stat
from wand import image as wand_image


def average(lst):
    """Helper function to get average of a list."""
    return sum(lst) / len(lst)


def hash_from_file_name(item: Path):
    """Returns just the hash portion of a files name"""
    if item.name.find("(") > 0:
        hash_str = re.sub(r" \(\d*\)", "", item.stem)
    else:
        hash_str = item.stem

    return hash_str


def hash_video(area: Path, _: int) -> bytearray:
    """Returns hash of a video file"""
    area = area.open("rb")
    read_bytes: bytes = area.read(1024)
    hash_str = sha256(read_bytes.hex("-", 16).encode("UTF-8"))
    return bytearray(hash_str.digest())


def png_of_heic(heic_file: Path) -> PIL.Image.Image:
    """Creates and returns a PNG version of an HEIC file"""
    path = str(heic_file)
    png_file = TemporaryFile()

    with wand_image.Image(filename=path) as img:
        print(path + ": converting....")
        img.format = 'png'
        img.save(file=png_file)
        print(path + ": converted")

    return PIL.Image.open(png_file)


def hash_heic(area: Union[PIL.Image.Image, Path], depth: int) -> bytearray:
    """Hash function of HEIC"""
    if isinstance(area, Path):
        png_file = png_of_heic(area)
    else:
        png_file = area

    ret_val = hash_helper(png_file, depth)

    png_file.close()
    return ret_val


def hash_helper(area: Path, depth: int) -> Union[None, bytearray]:
    """Recursively hash to the desired depth."""
    path = None
    if isinstance(area, Path):
        path = area
        area = PIL.Image.open(area)
    byte_array = bytearray()

    if depth == 0:
        return None

    width = area.width
    height = area.height
    avg = Stat(area).mean[:3]
    byte = int(average(avg)).to_bytes(1, "big")
    byte_array += byte
    for band in avg:
        byte_array += int(band).to_bytes(1, "big")

    if depth == 1:
        return byte_array

    boxes = [
        (0.0 * width, 0.0 * height, 0.5 * width, 0.5 * height),
        (0.5 * width, 0.0 * height, 1.0 * width, 0.5 * height),
        (0.0 * width, 0.5 * height, 0.5 * width, 1.0 * height),
        (0.5 * width, 0.5 * height, 1.0 * width, 1.0 * height),
    ]

    sub_arrays = []
    for box in boxes:
        sub_arrays.append(hash_helper(area.crop(box), depth - 1))

    sub_arrays.sort(key=lambda x: x[0])

    for arr in sub_arrays:
        byte_array += arr

    if path is not None:
        area.close()
    return byte_array


strategy_map = {
    ".jpg": hash_helper,
    ".PNG": hash_helper,
    ".JPG": hash_helper,
    ".jpeg": hash_helper,
    ".JPEG": hash_helper,
    ".png": hash_helper,
    ".WEBP": hash_helper,
    ".HEIC": hash_heic,
    ".MOV": hash_video,
    ".mov": hash_video,
    ".MP4": hash_video,
    ".mp4": hash_video,
    ".AAE": hash_video,
}


class HashedImages:
    """Class used to easily access hashed images."""
    def __init__(self):
        self._root_folder = Path("G:\\Hashed Pictures")
        self.all_hashed_files: list[Path] = self._get_files_rec(self._root_folder)
        print(len(self.all_hashed_files))

    def hash_and_size_list(self) -> dict[str, set[int]]:
        """Returns a dictionary with the file sizes as values and hashes as keys. Allows
        collisions to be found and resolved."""
        fil: Path
        ret_dict = {(key, {fil.stat().st_size for fil in value}) for key, value in self.hashes}
        return ret_dict

    def _get_files_rec(self, path: Path) -> list:
        """ Given a file path returns a list of files within that path recursively """
        if path.is_file():
            return [path]

        current_files = []
        for item in path.iterdir():
            current_files += self._get_files_rec(item)

        return current_files

    @staticmethod
    def hash_from_file_name(file: Path):
        """Returns just the hash portion of a files name"""
        if file.name.find("(") > 0:
            return re.sub(r" \(\d*\)", "", file.stem)
        return file.stem

    @staticmethod
    def is_image(file: Path):
        return strategy_map[file.suffix] is not hash_video

    @staticmethod
    def deep_hash(file: Path):
        """Returns a deeper hash of an image for comparison purposes."""
        suffix = file.suffix
        hash_strategy = strategy_map[suffix]
        return hash_strategy(file, 5)

    @cache
    def hashes(self) -> dict[str, list[Path]]:
        """A dictionary of hashes and their paths"""
        ret_dict = {}
        for file in self.all_hashed_files:
            hash_string = self.hash_from_file_name(file)
            if hash_string not in ret_dict:
                ret_dict[hash_string] = []

            ret_dict[hash_string].append(file)
        return ret_dict

    def all_collisions(self) -> dict[str, list[Path]]:
        """Returns a list of hashes that may have collisions"""
        ret_dict = {key: value for key, value in self.hashes().items() if len(value) > 1}
        return ret_dict
