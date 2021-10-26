from hashing import hash_helper, strategy_map
from PIL.Image import Image
from PIL import Image
from pathlib import Path
from shutil import copyfile
from multiprocessing import Pool
import re

source = Path("G:\\Camera Roll to Hash")
dest = Path("G:\\Hashed Pictures")




def hash_item(item: Path):
    if item.suffix not in strategy_map:
        return
    
    hashed = strategy_map[item.suffix](item, 2).hex("-", 4)
    return hashed


def hash_and_move(item: Path):
    hashed = hash_item(item)
    if hashed is None:
        return
    
    new_file = dest.joinpath(f"{hashed}{item.suffix}")
    copy = 1
    while new_file.exists():
        new_file = dest.joinpath(f"{hashed} ({copy}){item.suffix}")
        copy += 1
    
    move(item, new_file)

def move(item: Path, new_file: Path):
    try:
        item.rename(new_file)
    except PermissionError as e:
        print(f"{e}")
        pass

if __name__ == '__main__':
    pool = Pool(6)
    
    pool.map(hash_and_move, source.iterdir())
    
    file_dict = {}
    
    for item in dest.iterdir():
        if item.name.find("(") > 0:
            hash = re.sub(" \(\d*\)", "", item.stem)
            print(hash)
        else:
            hash = item.stem
            print(hash)
        
        if hash not in file_dict:
            file_dict[hash] = []
            
        file_dict[hash].append(item)
    
    dups = [lis for lis in file_dict.values() if len(lis) > 1]
    path: Path
    for dup in dups:
        size_set = set()
        for path in dup:
            if path.stat().st_size in size_set:
                path.rename(f"G:\\ToBeDeleted\\{path.name}")
            else:
                size_set.add(path.stat().st_size)
    
    print(dups)
