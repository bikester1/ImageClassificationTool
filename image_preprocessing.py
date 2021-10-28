"""
This module contains utility functions related to image processing.
"""
from pathlib import Path

import numpy as np
from wand.color import Color
from wand.image import Image


def load_image(file: Path) -> Image:
    """
    Loads image as Image
    :param file: path to image
    :return: Image of file
    """
    img = Image(filename=file)
    img.type = "grayscale"
    width = 256
    height = 256
    img.transform(resize=f"{width}x{height}")

    border_width = (width + 2 - img.width) // 2
    border_height = (height + 2 - img.height) // 2

    img.border(color=Color('black'), width=border_width, height=border_height)

    img.sample(width=width, height=height)
    return img


def load_image_as_256_array(file: Path) -> np.array:
    """
    Loads file path as 512x512x1 np array.
    Closes Wand Image of file to save memory and only maintain the np array version
    :param file: path to image
    :return: 512x512x1 np array of image
    """
    img = load_image(file)
    ret_val = np.array(img)
    img.close()
    return ret_val
