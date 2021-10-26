"""Moves files based on if they are duplicates."""
import os
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from find_duplicates import hash_item
from hashing import hash_from_file_name


def all_images() -> dict:
    """Returns a dict of file names and file paths"""
    root_folder = Path("G:\\Hashed Pictures")
    all_files = {
                     this_file.name: this_file
                     for sub_dir in root_folder.iterdir()
                     for this_file in sub_dir.iterdir()
                 }
    return all_files


def existing_hashes() -> dict:
    """Returns a dict of hash strings and files that have duplicates."""
    root_folder = Path("G:\\Hashed Pictures")
    hashes = {}
    for fold in root_folder.iterdir():
        if fold.is_dir():
            for item in fold.iterdir():
                if item.name.find("(") > 0:
                    hash_string = re.sub(r" \(\d*\)", "", item.stem)
                else:
                    hash_string = item.stem

                if hash_string not in hashes:
                    hashes[hash_string] = []

                if hash_string in hashes[hash_string]:
                    item.rename(f"G:\\ToBeDeleted\\{item.name}")
                else:
                    hashes[hash_string].append(item.stat().st_size)
    return hashes


def is_copied(lookup: dict, item: Path):
    """If the item exists already check if it is the same size."""
    try:
        item_hash = hash_item(item)
    except Exception as exception:
        print(exception)
        return False
    if item_hash not in lookup:
        print(f"Item not copied: {item}")
    elif item.stat().st_size not in lookup[item_hash]:
        print(f"Item not copied: {item}")
    else:
        print(f"Item copied: {item}")
        item.rename(f"G:\\ToBeDeleted\\{item.name}")
    return item_hash in lookup


def main():
    """Run if __name__ is __main__"""
    pool = Pool(6)
    folder = Path("G:\\Hashed Pictures")

    for file in folder.iterdir():
        if not file.is_file():
            continue

        new_folder = folder.joinpath(f"{file.name[:2]}")
        if not new_folder.exists():
            os.makedirs(new_folder)

        new_file = new_folder.joinpath(file.name)
        hashed = hash_from_file_name(new_file)
        copy_number = 1
        while new_file.exists():
            new_file = new_folder.joinpath(f"{hashed} ({copy_number}){new_file.suffix}")
            copy_number += 1
        file.rename(new_folder.joinpath(new_file.name))

    all_hashes = existing_hashes()

    check_folder = Path("G:\\Sierra Iphone 11 Picture Backup 09-26-21 - Copy")

    copy_func = partial(is_copied, all_hashes)
    pool.map(copy_func, check_folder.iterdir())


if __name__ == '__main__':
    main()
