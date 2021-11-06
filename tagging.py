"""This module handles the tagging of images."""
import json
from collections import Counter
from pathlib import Path

from hashing import HashedImages


class ImageTagging:
    """Tagging object used to access tagged image sets from a JSON file."""
    def __init__(self):
        self.json_file_path = Path("G:\\saved_tags.json")
        self.json_file = open(self.json_file_path, "a+", encoding="utf-8")
        self.json_file.seek(0)
        self.all_images = HashedImages().all_hashed_files
        self.image_iter = self.all_images.__iter__()
        self.tagged_images = {}
        self._load_tags()
        self.current_image = Path()

        self.tag_counts = Counter([tag for fil_name, dat in self.tagged_images.items()
                                   for tag in dat["Tags"]])

        self.output_tags = [tag for tag, count in self.tag_counts.most_common(10)]
        if len(self.output_tags) == 0:
            self.output_tags = ["screenshot"]

        self.output_tags = ["dog", "duke", "person", "sierra"]

        self.file_extensions = {
            ".png",
            ".heic",
            ".PNG",
            ".HEIC",
            ".JPG",
            ".jpg",
            ".jpeg",
            ".JPEG",
        }

    def all_tags(self) -> set:
        """
        :return: Set of all tags in the tag JSON file.
        """
        tag_lists = [file["Tags"] for file in self.tagged_images.values()]
        return {tag for file in tag_lists for tag in file}

    def files_from_tag_list(self, tags: list[str]) -> list[Path]:
        """Takes a list of strings and looks for file paths that have one or more associated tags.

        :param tags: List of tags for which to find matching images.
        :return: List of paths to images that are tagged with one ore more of the specified tags.
        """
        tag_set = set(tags)

        ret_list = []
        for file, dat in self.tagged_images.items():
            if len(set(dat["Tags"]) & tag_set):
                ret_list.append(file)

        return self.files_from_file_name_list(ret_list)

    def files_from_file_name_list(self, file_names: list[str]) -> list[Path]:
        """Hashed images are stored in a hierarchical file structure. This method will tage a
        list of image names and find the associated paths for them.

        :param file_names: List of file name hashes.
        :return: List of paths to individual images.
        """
        return [fil for fil in self.all_images if fil.name in file_names]

    def _load_tags(self):
        """Loads tags from JSON file."""
        try:
            self.tagged_images = json.load(self.json_file)
        except ValueError:
            self.tagged_images = {}

    def save_tags(self):
        """Saves tags to JSON file."""
        self.json_file.truncate(0)
        self.json_file.seek(0)
        json.dump(self.tagged_images, self.json_file, indent=4)
        self.json_file.flush()

    def get_next_image(self) -> Path:
        """Returns path of the next image that is not already tagged."""
        next_image = self.image_iter.__next__()

        while next_image.name in self.tagged_images \
                or next_image.suffix not in self.file_extensions:
            next_image = self.image_iter.__next__()
            print(next_image.name)

        return next_image

    def tag_image(self, img_name: str, tags: list[str]):
        """Applies a list of tags to a specific image.

        :param img_name: Hashed name of image.
        :param tags: Tags associated with image.
        :return: None.
        """
        img_dict = {
            "Tags": tags,
        }
        self.tagged_images[img_name] = img_dict
        self.save_tags()

    def file_tags(self, path: Path) -> list[str]:
        """This method takes a path and gives a list of that files tags.

        :param path: Path to file for which to find tags
        :return: List of tags associated with the given file
        """
        return self.tagged_images[path.name]["Tags"]


tagging = ImageTagging()
