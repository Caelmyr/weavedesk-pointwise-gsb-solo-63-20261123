from io import BytesIO

import numpy as np
from PIL import Image

from .model import InputError
from .photos import png_bytes


def stored_mask(blobs, mask):
    image = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    return blobs.put(png_bytes(image))


def restored_mask(blobs, digest, extent):
    with Image.open(BytesIO(blobs.get(digest))) as source:
        values = np.asarray(source.convert("L"), dtype=np.uint8)
    width, height = int(extent[0]), int(extent[1])
    if values.shape[::-1] != (width, height):
        values = np.asarray(Image.fromarray(values).resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8)
    return values >= 128


def stored_plane(blobs, values):
    stream = BytesIO()
    np.save(stream, np.asarray(values, dtype=np.float16), allow_pickle=False)
    return blobs.put(stream.getvalue())


def restored_plane(blobs, digest, bounds):
    with BytesIO(blobs.get(digest)) as stream:
        values = np.asarray(np.load(stream, allow_pickle=False), dtype=np.float32)
    width, height = int(bounds[2]), int(bounds[3])
    if values.shape[:2] != (height, width):
        raise InputError("The approved reference plane no longer matches this specimen plane. Approve it again.")
    return values
