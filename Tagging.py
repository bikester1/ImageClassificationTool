from Hashing import *
from collections import Counter

import json


class ImageTagging:
    
    def __init__(self):
        self.json_file_path = Path("G:\\saved_tags.json")
        self.json_file = open(self.json_file_path, "a+")
        self.json_file.seek(0)
        self.all_images = HashedImages().all_hashed_files
        self.image_iter = self.all_images.__iter__()
        self.tagged_images = {}
        self.load_tags()
        self.current_image = Path()
        
        self.tag_counts = Counter([tag for fil_name, dat in self.tagged_images.items() for tag in dat["Tags"]])
        
        self.output_tags = [tag for tag, count in self.tag_counts.most_common(10)]
        if not len(self.output_tags):
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
        tag_lists = [file["Tags"] for file in self.tagged_images.values()]
        return set([tag for file in tag_lists for tag in file])
    
    def files_from_tag_list(self, tags: list[str]) -> list[Path]:
        tag_set = set(tags)
        
        ret_list = []
        for file, dat in self.tagged_images.items():
            if len(set(dat["Tags"]) & tag_set):
                ret_list.append(file)
        
        return self.files_from_file_name_list(ret_list)
    
    def files_from_file_name_list(self, file_names: list[str]) -> list[Path]:
        return [fil for fil in self.all_images if fil.name in file_names]
    
    def load_tags(self):
        try:
            self.tagged_images = json.load(self.json_file)
        except ValueError:
            self.tagged_images = {}
    
    def save_tags(self):
        self.json_file.truncate(0)
        self.json_file.seek(0)
        json.dump(self.tagged_images, self.json_file, indent=4)
        self.json_file.flush()
    
    def get_next_image(self) -> Path:
        next_image = self.image_iter.__next__()
        
        while next_image.name in self.tagged_images or next_image.suffix not in self.file_extensions:
            next_image = self.image_iter.__next__()
            print(next_image.name)
        
        return next_image
    
    def tag_image(self, img_name: str, tags: list[str]):
        img_dict = {
            "Tags": tags,
        }
        self.tagged_images[img_name] = img_dict
        self.save_tags()

