from pathlib import Path
from PIL import Image


def load_local_image(image_path):

    image_path = Path(image_path)

    image = Image.open(image_path)

    return image.convert("RGB")
