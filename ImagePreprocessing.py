from pathlib import *
from wand.color import Color
from wand.image import *

import numpy as np


def load_image(file: Path) -> Image:
    img = Image(filename=file)
    img.type = "grayscale"
    width = 512
    height = 512
    img.transform(resize=f"{width}x{height}")

    border_width = (width + 2 - img.width) // 2
    border_height = (height + 2 - img.height) // 2

    img.border(color=Color('black'), width=border_width, height=border_height)
    
    img.sample(width=width, height=height)
    return img


def load_image_as_512_array(file: Path) -> np.array:
    img = load_image(file)
    ret_val = np.array(img)
    img.close()
    return ret_val
