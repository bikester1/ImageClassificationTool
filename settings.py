import json
import pathlib
import os

class SettingController:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SettingController, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.settings_file = pathlib.Path("settings.json")

        if not self.settings_file.exists():
            with open(self.settings_file, "x") as new_file:
                json.dump(self.defaults(), new_file)

        with open(self.settings_file, "r+") as fp:
            self.json: dict = json.load(fp)
            missing_settings = set(self.defaults()).difference(set(self.json.keys()))

            for setting in missing_settings:
                self.json[setting] = self.defaults()[setting]

            fp.truncate(0)
            fp.seek(0)
            json.dump(self.json, fp)

    def set_setting(self, setting: str, value: any):
        self.json[setting] = value
        with open(self.settings_file, "w+") as fp:
            json.dump(self.json, fp)

    def get_setting(self, setting) -> any:
        return self.json[setting]

    def defaults(self):
        return {
            "hashed_images_root": "C:\\",
            "image_source_folder": "C:\\",
        }
