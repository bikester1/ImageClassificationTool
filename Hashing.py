import os
import re
from functools import cache
from hashlib import sha256
from multiprocessing import *
from multiprocessing.connection import *
from pathlib import Path
from typing import Union

from PIL import Image
from PIL.Image import Image
from PIL.ImageStat import Stat
from wand import image as wndImage


def average(lst):
    return sum(lst) / len(lst)


def hash_from_file_name(item: Path):
    if item.name.find("(") > 0:
        hash = re.sub(" \(\d*\)", "", item.stem)
    else:
        hash = item.stem
    
    return hash


def hash_video(area: Union[Image, Path], depth: int) -> bytearray:
    if isinstance(area, Path):
        area = area.open("rb")
        bytes: bytes = area.read(1024)
        hash = sha256(bytes.hex("-", 16).encode("UTF-8"))
    return bytearray(hash.digest())


def png_of_heic(heic_file: Path) -> Path:
    path = str(heic_file)
    png_file = Path(path + ".png")
    
    if png_file.exists():
        return png_file
    else:
        with wndImage.Image(filename=path) as img:
            print(path + ": converting....")
            img.format = 'png'
            img.save(filename=(path + ".png"))
            print(path + ": converted")
    
    return png_file


def hash_heic(area: Union[Image, Path], depth: int) -> bytearray:
    if isinstance(area, Path):
        png_file = png_of_heic(area)
        
    ret_val = hash_helper(png_file, depth)
    
    os.remove(png_file)
    return ret_val


def hash_helper(area: Union[Image, Path], depth: int) -> bytearray:
    path = None
    if isinstance(area, Path):
        path = area
        area = Image.open(path)
    byte_array = bytearray()
    
    if depth == 0:
        return
    
    width = area.width
    height = area.height
    avg = Stat(area).mean[:3]
    sum = 0
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
    
    subarrs = []
    for box in boxes:
        subarrs.append(hash_helper(area.crop(box), depth - 1))
    
    subarrs.sort(key=lambda x: x[0])
    
    for arr in subarrs:
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
    
    def __init__(self):
        self._root_folder = Path("G:\\Hashed Pictures")
        self.parent_connection,child_connection = Pipe()
        self.image_loading_process = Process()
        self.all_hashed_files: list[Path]= self._get_files_rec(self._root_folder)
        print(len(self.all_hashed_files))
    
    @cache
    def hash_and_size_list(self) -> dict:
        ret_dict = {}
        for file in self.all_hashed_files:
            hash = self.hash_from_file_name(file)
            if hash not in ret_dict:
                ret_dict[hash] = set()
            
            ret_dict[hash].add(file.stat().st_size)
        return ret_dict
        
    def _get_files_rec(self, path: Path) -> list:
        """ Given a file path returns a list of files within that path recursively """
        if path.is_file():
            return [path]
        
        current_files = []
        for item in path.iterdir():
            current_files += self._get_files_rec(item)
        
        return current_files

    def hash_from_file_name(self, file: Path):
        if file.name.find("(") > 0:
            return re.sub(" \(\d*\)", "", file.stem)
        else:
            return file.stem
        
        return
        
