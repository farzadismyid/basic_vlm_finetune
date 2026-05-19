from PIL import Image
import requests


def load_image_from_url(url):

    image = Image.open(
        requests.get(url, stream=True).raw
    )

    return image.convert("RGB")
