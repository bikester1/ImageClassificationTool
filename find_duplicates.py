"""This module looks for duplicate images by hashing and then moveing duplicates to a ready to
delete folder."""
import re
from multiprocessing import Pool
from pathlib import Path

from hashing import strategy_map

source = Path("G:\\Camera Roll to Hash")
destination = Path("G:\\Hashed Pictures")


def hash_item(file_path: Path):
    """Returns a hash from a path."""
    if file_path.suffix not in strategy_map:
        return None

    hashed = strategy_map[file_path.suffix](file_path, 2).hex("-", 4)
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


if __name__ == '__main__':
    pool = Pool(6)

    pool.map(hash_and_move, source.iterdir())

    file_dict = {}

    for item in destination.iterdir():
        if item.name.find("(") > 0:
            hash_string = re.sub(r" \(\d*\)", "", item.stem)
            print(hash_string)
        else:
            hash_string = item.stem
            print(hash_string)

        if hash_string not in file_dict:
            file_dict[hash_string] = []

        file_dict[hash_string].append(item)

    duplicates = [lis for lis in file_dict.values() if len(lis) > 1]
    path: Path
    for dup in duplicates:
        size_set = set()
        for path in dup:
            if path.stat().st_size in size_set:
                path.rename(f"G:\\ToBeDeleted\\{path.name}")
            else:
                size_set.add(path.stat().st_size)

    print(duplicates)
