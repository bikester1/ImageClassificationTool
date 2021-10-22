from wand import image
from pathlib import Path


test: image.Image = image.Image(filename=Path("G:\\Hashed Pictures\\3e57352d-35472e28-3f5c352d-41513c35-43683729.PNG"))
test.metadata["MyTestTags"] = f"Hello lol"
test.save(filename="G:\\test.png")

test: image.Image = image.Image(filename="G:\\test.png")
print(list(test.metadata.items()))