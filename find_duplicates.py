"""This module looks for duplicate images by hashing and then moveing duplicates to a ready to
delete folder."""
import re
from multiprocessing import Pool
from pathlib import Path

from hashing import strategy_map


class DuplicateFinder:

    def __init__(self, source_dir: Path, destination_dir: Path, trash_dir: Path):
        self.source = source_dir
        self.destination = destination_dir
        self.trash = trash_dir

    def hash_item(self, file_path: Path):
        """Returns a hash from a path."""
        if file_path.suffix not in strategy_map:
            return None

        hashed = strategy_map[file_path.suffix](file_path, 2).hex("-", 4)
        return hashed


    def hash_and_move(self, file_path: Path):
        """Hashes image and then moves it to its final location"""
        hashed = self.hash_item(file_path)
        if hashed is None:
            return

        new_file = self.destination.joinpath(f"{hashed}{file_path.suffix}")
        copy = 1
        while new_file.exists():
            new_file = self.destination.joinpath(f"{hashed} ({copy}){file_path.suffix}")
            copy += 1

        self.move(file_path, new_file)


    def move(self, file_path: Path, new_file: Path):
        """Tries to move a file to a new file path."""
        try:
            file_path.rename(new_file)
        except PermissionError as exception:
            print(f"{exception}")


    def move_duplicates(self):
        pool = Pool(6)

        pool.map(self.hash_and_move, self.source.iterdir())

        file_dict = {}

        for item in self.destination.iterdir():
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
                    path.rename(f"{self.trash}\\{path.name}")
                else:
                    size_set.add(path.stat().st_size)

        print(duplicates)
