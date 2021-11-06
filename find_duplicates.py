"""This module looks for duplicate images by hashing and then moveing duplicates to a ready to
delete folder."""
import re
from multiprocessing import Pool
from pathlib import Path

from hashing import strategy_map, HashedImages

source = Path("G:\\Camera Roll to Hash")
destination = Path("G:\\Hashed Pictures")
deleted_file_path = Path("G:\\ToBeDeleted")


def hash_item(file_path: Path, depth: int = 2):
    """Returns a hash from a path."""
    if file_path.suffix not in strategy_map:
        return None

    hashed = strategy_map[file_path.suffix](file_path, depth).hex("-", 4)
    return hashed


def hash_and_move(file_path: Path):
    """Hashes image and then moves it to its final location"""
    hashed = hash_item(file_path)
    if hashed is None:
        return

    new_file = destination.joinpath(f"{hashed}{file_path.suffix}")
    copy = 1
    while new_file.exists():
        new_file = destination.joinpath(f"{hashed} ({copy}){file_path.suffix}")
        copy += 1

    move(file_path, new_file)


def move(file_path: Path, new_file: Path):
    """Tries to move a file to a new file path."""
    try:
        file_path.rename(new_file)
    except PermissionError as exception:
        print(f"{exception}")


def deep_hash_collisions(collisions: list[Path]) -> None:
    collision_dictionary: dict[str, list[Path]] = {}
    for pic in collisions:
        pic_hash = hash_item(pic, 5)
        if pic_hash not in collision_dictionary:
            collision_dictionary[pic_hash] = []
        collision_dictionary[pic_hash].append(pic)

    value: list[Path]
    key: str
    for key, value in collision_dictionary.items():
        min_size_image = min(value, key=lambda x: x.stat().st_size)
        value.remove(min_size_image)
        for img in value:
            if strategy_map[img.suffix] != strategy_map[".MOV"]:
                new_path = deleted_file_path.joinpath(f"{img.name}")
                move(img, new_path)

if __name__ == '__main__':
    pool = Pool(6)

    pool.map(hash_and_move, source.iterdir())

    hash_structure = HashedImages()
    collisions = hash_structure.all_collisions()
    pool.map(deep_hash_collisions, collisions.values())
