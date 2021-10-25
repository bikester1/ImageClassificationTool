from multiprocessing import Pool
from pathlib import *
import os
from FindDuplicates import hash_item
import re
from functools import partial
from Hashing import *


def all_images() -> dict:
    root_folder = Path(f"G:\\Hashed Pictures")
    all_files = dict([(file.name, file) for folder in root_folder.iterdir() for file in folder.iterdir()])
    return all_files

def existing_hashes() -> dict:
    root_folder = Path(f"G:\\Hashed Pictures")
    hashes = dict()
    for fold in root_folder.iterdir():
        if fold.is_dir():
            for item in fold.iterdir():
                if item.name.find("(") > 0:
                    hash = re.sub(" \(\d*\)", "", item.stem)
                else:
                    hash = item.stem

                if hash not in hashes:
                    hashes[hash] = []
                
                if hash in hashes[hash]:
                    item.rename(f"G:\\ToBeDeleted\\{item.name}")
                else:
                    hashes[hash].append(item.stat().st_size)
    return hashes


def is_copied(lookup: set, item: Path):
    try:
        item_hash = hash_item(item)
    except Exception as e:
        print(e)
        return False
    if item_hash not in lookup:
        print(f"Item not copied: {item}")
    elif item.stat().st_size not in lookup[item_hash]:
        print(f"Item not copied: {item}")
    else:
        print(f"Item copied: {item}")
        item.rename(f"G:\\ToBeDeleted\\{item.name}")
    return item_hash in lookup

    

if __name__ == '__main__':
    
    all_hashed_images = all_images()
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
        copy = 1
        while new_file.exists():
            new_file = new_folder.joinpath(f"{hashed} ({copy}){new_file.suffix}")
            copy += 1
        file.rename(new_folder.joinpath(new_file.name))

    all_hashes = existing_hashes()
    
    check_folder = Path("G:\\Sierra Iphone 11 Picture Backup 09-26-21 - Copy")
    
    copy_func = partial(is_copied, all_hashes)
    pool.map(copy_func, check_folder.iterdir())
    
