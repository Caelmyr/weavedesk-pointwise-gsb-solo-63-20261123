"""One physical redaction cover shared by every image in an evidence packet.

The cover is a single continuous 18 × 22 mm rectangle measured on the
registered specimen plane.  Every capture attached to a specimen is registered
to that same plane, so one cover position projects to the corresponding place
on the front-on and oblique captures, on report pictures derived from either,
and on the attached original files.  The payload accepts one centre only: a
packet cannot grow a separate cover position for each photograph.
"""

from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from .model import InputError, dictionary, numeric_pair, scalar
from .photos import png_bytes


COVER_WIDTH_MM = 18.0
COVER_HEIGHT_MM = 22.0
COVER_FILL = (27, 32, 34)
DIMENSION_TOLERANCE = 1e-6
EDGE_TOLERANCE = 1e-6
QUAD_TOLERANCE = 0.02
COVER_KEYS = {
    "enabled", "applied", "width", "height", "width_mm", "height_mm", "size", "size_mm",
    "center", "centre", "position", "origin", "center_mm", "x", "y", "x_mm", "y_mm",
    "center_px", "center_plane", "angle", "angle_deg", "rotation", "rotation_deg",
}


def _centimetre_pair(data, key):
    if key not in data:
        return None
    return numeric_pair(data.get(key), "Cover centre", lower=-1e6, upper=1e6)


def cover_centre(data, tile):
    """The one cover centre, as a fraction of the registered plane."""
    if (point := _centimetre_pair(data, "center_mm")) is not None:
        return [point[0] / float(tile["width_mm"]), point[1] / float(tile["height_mm"])]
    if "x_mm" in data or "y_mm" in data:
        if "x_mm" not in data or "y_mm" not in data:
            raise InputError("A cover position needs both an X and a Y coordinate in millimetres.")
        return [
            scalar(data["x_mm"], "Cover X", lower=-1e6, upper=1e6) / float(tile["width_mm"]),
            scalar(data["y_mm"], "Cover Y", lower=-1e6, upper=1e6) / float(tile["height_mm"]),
        ]
    if (point := _centimetre_pair(data, "center_px")) is not None or (point := _centimetre_pair(data, "center_plane")) is not None:
        width, height = tile["plane"]["size"]
        return [point[0] / float(width), point[1] / float(height)]
    for key in ("center", "centre", "position", "origin"):
        if (point := _centimetre_pair(data, key)) is not None:
            return point
    if "x" in data or "y" in data:
        if "x" not in data or "y" not in data:
            raise InputError("A cover position needs both an X and a Y coordinate.")
        return [
            scalar(data["x"], "Cover X", lower=-1, upper=2),
            scalar(data["y"], "Cover Y", lower=-1, upper=2),
        ]
    raise InputError("Choose one cover centre on the registered specimen plane.")


def cover_angle(data):
    for key in ("angle_deg", "angle", "rotation_deg", "rotation"):
        if key in data:
            return scalar(data[key], "Cover rotation", lower=-360, upper=360)
    return 0.0


def rectangle_corners(centre_mm, width_mm, height_mm, angle_deg):
    radians = math.radians(angle_deg)
    cosine, sine = math.cos(radians), math.sin(radians)
    edges = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    return [
        [
            centre_mm[0] + (sx * width_mm / 2 * cosine - sy * height_mm / 2 * sine),
            centre_mm[1] + (sx * width_mm / 2 * sine + sy * height_mm / 2 * cosine),
        ]
        for sx, sy in edges
    ]


def cover_spec(payload, tile):
    """Validate the one permitted cover and describe it in plane coordinates."""
    data = dictionary(payload, "Redaction cover")
    if data.get("enabled") is False or data.get("applied") is False:
        return None
    if set(data) - COVER_KEYS:
        raise InputError("A packet carries one cover size and one cover position; extra cover fields are not allowed.")
    width = next((scalar(data[key], "Cover width", lower=0.001, upper=100000) for key in ("width_mm", "width") if key in data), None)
    height = next((scalar(data[key], "Cover height", lower=0.001, upper=100000) for key in ("height_mm", "height") if key in data), None)
    if width is None and height is None and ("size" in data or "size_mm" in data):
        width, height = numeric_pair(data.get("size_mm", data.get("size")), "Cover size", lower=0.001, upper=100000)
    if width is None or height is None:
        raise InputError("Give the cover size in millimetres.")
    if not any(
        abs(width - left) <= DIMENSION_TOLERANCE and abs(height - top) <= DIMENSION_TOLERANCE
        for left, top in ((COVER_WIDTH_MM, COVER_HEIGHT_MM), (COVER_HEIGHT_MM, COVER_WIDTH_MM))
    ):
        raise InputError("The evidence cover must measure 18 × 22 mm (either axis order).")
    thesis = tile.get("width_mm"), tile.get("height_mm")
    if None in thesis or min(float(value) for value in thesis) <= 0:
        raise InputError("Set the specimen width and height in millimetres before adding a measured cover.")
    if not tile.get("plane") or not tile["plane"].get("size"):
        raise InputError("Set a registered specimen plane before adding a measured cover.")
    physical_width, physical_height = (float(value) for value in thesis)
    centre = cover_centre(data, tile)
    angle = cover_angle(data)
    corners = rectangle_corners(
        [centre[0] * physical_width, centre[1] * physical_height], width, height, angle,
    )
    normalized = [[u / physical_width, v / physical_height] for u, v in corners]
    if any(u < -EDGE_TOLERANCE or u > 1 + EDGE_TOLERANCE or v < -EDGE_TOLERANCE or v > 1 + EDGE_TOLERANCE
           for u, v in normalized):
        raise InputError("The cover must sit entirely inside the registered specimen plane.")
    plane_width, plane_height = tile["plane"]["size"]
    return {
        "shape": "rectangle",
        "continuous": True,
        "width_mm": round(float(width), 6),
        "height_mm": round(float(height), 6),
        "angle_deg": round(float(angle), 6),
        "center": [round(float(centre[0]), 9), round(float(centre[1]), 9)],
        "plane_size": [int(plane_width), int(plane_height)],
        "plane_corners": [[round(u, 9), round(v, 9)] for u, v in normalized],
        "fill_rgb": list(COVER_FILL),
    }


def cover_summary(spec, members):
    summary = {
        key: spec[key]
        for key in (
            "shape", "continuous", "width_mm", "height_mm", "angle_deg", "center",
            "plane_size", "plane_corners", "fill_rgb",
        )
    }
    summary["applied"] = True
    summary["members"] = members
    return summary


def plane_polygon(spec, size):
    width, height = size
    return [(float(u) * width - 0.5, float(v) * height - 0.5) for u, v in spec["plane_corners"]]


def projected_quad(transform, spec):
    """Project the plane cover through a capture registration to photo coordinates."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.size == 9:
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise InputError("A capture has no usable registration for the evidence cover.")
    quad = []
    for u, v in spec["plane_corners"]:
        denominator = matrix[2, 0] * u + matrix[2, 1] * v + matrix[2, 2]
        if abs(denominator) < 1e-12:
            raise InputError("The evidence cover cannot be projected onto an attached photograph.")
        x = (matrix[0, 0] * u + matrix[0, 1] * v + matrix[0, 2]) / denominator
        y = (matrix[1, 0] * u + matrix[1, 1] * v + matrix[1, 2]) / denominator
        if not np.isfinite(x) or not np.isfinite(y):
            raise InputError("The evidence cover cannot be projected onto an attached photograph.")
        quad.append([round(float(x), 9), round(float(y), 9)])
    return quad


def pixel_quad(quad, size):
    width, height = size
    return [(x * width - 0.5, y * height - 0.5) for x, y in quad]


def paste_cover(image, points):
    """Paint an opaque, untextured cover onto an image."""
    working = image.convert("RGBA")
    layer = Image.new("RGBA", working.size, (*COVER_FILL, 255))
    stencil = Image.new("L", working.size, 0)
    ImageDraw.Draw(stencil).polygon(points, fill=255)
    working.paste(layer, (0, 0), stencil)
    return working


def apply_plane_cover(image, spec):
    return paste_cover(image, plane_polygon(spec, image.size))


def redact_original(raw, spec, transform):
    """Return a covered PNG copy of an original capture, its photo quad, and size."""
    try:
        with Image.open(BytesIO(raw)) as source:
            upright = ImageOps.exif_transpose(source)
            upright.load()
            quad = projected_quad(transform, spec)
            covered = paste_cover(upright, pixel_quad(quad, upright.size))
            return png_bytes(covered), quad, list(upright.size)
    except InputError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        raise InputError("An attached original could not be decoded for the evidence cover.") from error


def validated_quad(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise InputError(f"{label} needs four projected corners.")
    quad = []
    for point in value:
        pair = numeric_pair(point, label, lower=-0.5, upper=1.5)
        if not all(-QUAD_TOLERANCE <= coordinate <= 1 + QUAD_TOLERANCE for coordinate in pair):
            raise InputError(f"{label} leaves the photograph.")
        quad.append([round(pair[0], 9), round(pair[1], 9)])
    return quad


def solid_cover(image, quad, fill, tolerance=6):
    """Fraction of the declared cover interior that is opaque, untextured fill."""
    working = image.convert("RGB")
    width, height = working.size
    stencil = Image.new("L", working.size, 0)
    ImageDraw.Draw(stencil).polygon([(x * width - 0.5, y * height - 0.5) for x, y in quad], fill=255)
    mask = np.asarray(stencil, dtype=bool)
    if not mask.any():
        return 0.0
    interior = ndimage.binary_erosion(mask, iterations=2, border_value=0)
    if not interior.any():
        interior = mask
    pixels = np.asarray(working, dtype=np.int16)[interior]
    wanted = np.asarray(fill, dtype=np.int16)
    return float(np.all(np.abs(pixels - wanted) <= tolerance, axis=1).mean())
