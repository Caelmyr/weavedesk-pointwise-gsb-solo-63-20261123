from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .model import InputError, MAX_REGISTERED_DIM, collection, numeric_pair, physical_scale
from .photos import open_working, linear_to_srgb, srgb_to_linear


DEFAULT_CORNERS = [[0, 0], [1, 0], [1, 1], [0, 1]]
MIN_OVERLAP = 0.08
WARP_ROW_BLOCK = 128
MIN_SUPPORT_ALPHA = 0.995


@dataclass(frozen=True)
class SampleFrame:
    linear: np.ndarray
    valid: np.ndarray
    photo: dict
    transform: tuple
    scale: tuple | None
    clipped: np.ndarray

    @property
    def extent(self):
        height, width = self.valid.shape
        return width, height

    def display(self):
        return linear_to_srgb(self.linear)


def oriented_cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def signed_area(vertices):
    linked = list(vertices[1:]) + [vertices[0]]
    return 0.5 * sum(
        first[0] * second[1] - first[1] * second[0]
        for first, second in zip(vertices, linked)
    )


def on_segment(a, b, p):
    tiny = 1e-9
    if abs(oriented_cross(a, b, p)) > tiny:
        return False
    return (
        min(a[0], b[0]) - tiny <= p[0] <= max(a[0], b[0]) + tiny
        and min(a[1], b[1]) - tiny <= p[1] <= max(a[1], b[1]) + tiny
    )


def segments_intersect(a, b, c, d):
    ab_at_c = oriented_cross(a, b, c)
    ab_at_d = oriented_cross(a, b, d)
    cd_at_a = oriented_cross(c, d, a)
    cd_at_b = oriented_cross(c, d, b)
    if ab_at_c * ab_at_d < 0 and cd_at_a * cd_at_b < 0:
        return True
    return any((on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b)))


def validate_polygon(vertices):
    if abs(signed_area(vertices)) < 0.0005:
        raise InputError("Region needs an area larger than 0.05% of the tile image.")
    size = len(vertices)
    seen = {tuple(vertex) for vertex in vertices}
    if len(seen) != size:
        raise InputError("A polygon cannot use the same vertex more than once.")
    for index, a in enumerate(vertices):
        b = vertices[(index + 1) % size]
        for other_index in range(index + 2, size):
            if index == 0 and other_index == size - 1:
                continue
            c = vertices[other_index]
            d = vertices[(other_index + 1) % size]
            if segments_intersect(a, b, c, d):
                raise InputError("Region edges cannot cross.")
    return vertices


def corners_parameter(value):
    points = [
        numeric_pair(item, "Photo corner", lower=0, upper=1)
        for item in collection(value, "Photo corners", minimum=4, maximum=4)
    ]
    if signed_area(points) < MIN_OVERLAP:
        raise InputError("Pick a clockwise tile plane that covers at least 8% of the photo.")
    for index in range(4):
        if oriented_cross(points[index - 2], points[index - 1], points[index]) <= 0.002:
            raise InputError("Tile corners must form a convex clockwise quadrilateral without a folded edge.")
    return points


def canvas_dimensions(photo, corners, physical_width=None, physical_height=None):
    points = np.asarray(corners, dtype=np.float64) * [photo["width"], photo["height"]]
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    apparent_width = float(np.mean(lengths[::2]))
    apparent_height = float(np.mean(lengths[1::2]))
    if min(apparent_width, apparent_height) < 16:
        raise InputError("Marked tile is too small to create a usable specimen plane.")
    if physical_width and physical_height:
        density = min(apparent_width / physical_width, apparent_height / physical_height)
        apparent_width = density * physical_width
        apparent_height = density * physical_height
    longest = max(apparent_width, apparent_height)
    down = min(MAX_REGISTERED_DIM / longest, 1.0)
    dims = (int(round(apparent_width * down)), int(round(apparent_height * down)))
    if min(dims) < 16:
        raise InputError("The resulting specimen plane would have less than 16 pixels across an edge.")
    return dims


def homography(unit_target, source):
    x = np.asarray(unit_target, dtype=np.float64)
    y = np.asarray(source, dtype=np.float64)
    equations = np.zeros((8, 8), dtype=np.float64)
    rhs = y.reshape(-1)
    for index, ((tx, ty), (sx, sy)) in enumerate(zip(x, y)):
        equations[2 * index] = [tx, ty, 1, 0, 0, 0, -sx * tx, -sx * ty]
        equations[2 * index + 1] = [0, 0, 0, tx, ty, 1, -sy * tx, -sy * ty]
    try:
        values = np.linalg.solve(equations, rhs)
    except np.linalg.LinAlgError as error:
        raise InputError("The marked corners cannot define a stable image transform.") from error
    matrix = np.append(values, 1).reshape(3, 3)
    if (
        not np.all(np.isfinite(matrix))
        or np.linalg.cond(matrix) > 2000
        or np.min(np.c_[x, np.ones(4)] @ matrix[2]) <= 0.016
    ):
        raise InputError("The marked perspective is nearly singular. Choose a more front-facing photograph.")
    return matrix


def map_coordinates(matrix, target_x, target_y, source_width, source_height):
    denominator = matrix[2, 0] * target_x + matrix[2, 1] * target_y + matrix[2, 2]
    mapped_x = (matrix[0, 0] * target_x + matrix[0, 1] * target_y + matrix[0, 2]) / denominator
    mapped_y = (matrix[1, 0] * target_x + matrix[1, 1] * target_y + matrix[1, 2]) / denominator
    return mapped_x * source_width - 0.5, mapped_y * source_height - 0.5


def warp_capture(rgb, alpha, destination, transform):
    source_height, source_width = alpha.shape
    width, height = destination
    target_x = (np.arange(width, dtype=np.float32)[None, :] + 0.5) / width
    rgb_working = srgb_to_linear(rgb)
    weighted = rgb_working * alpha[:, :, None]
    src_clip = (np.max(rgb, axis=2) >= 253 / 255) | (np.max(rgb, axis=2) <= 5 / 255)
    linear = np.zeros((height, width, 3), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=bool)
    clipped = np.zeros((height, width), dtype=bool)
    for start in range(0, height, WARP_ROW_BLOCK):
        stop = min(start + WARP_ROW_BLOCK, height)
        target_y = (np.arange(start, stop, dtype=np.float32)[:, None] + 0.5) / height
        src_x, src_y = map_coordinates(transform, target_x, target_y, source_width, source_height)
        within = (src_x >= -0.5) & (src_x <= source_width - 0.5) & (src_y >= -0.5) & (src_y <= source_height - 0.5)
        coordinates = (src_y, src_x)
        opacity = ndimage.map_coordinates(alpha, coordinates, order=1, mode="nearest")
        present = within & (opacity >= MIN_SUPPORT_ALPHA)
        coverage[start:stop] = present
        clipped[start:stop] = (
            ndimage.map_coordinates(src_clip.astype(np.float32), coordinates, order=1, mode="nearest") > 0.025
        ) & present
        target = linear[start:stop]
        for channel in range(3):
            samples = ndimage.map_coordinates(weighted[:, :, channel], coordinates, order=1, mode="nearest")
            np.divide(samples, opacity, where=present, out=target[:, :, channel])
    return linear, coverage, clipped


def specimen_frame(blobs, tile, photo_id=None):
    plane = tile.get("plane")
    if plane is None:
        raise InputError("Set a base photo and mark the tile plane first.")
    identifier = photo_id or plane["photo_id"]
    photo = next((item for item in tile["photos"] if item["id"] == identifier), None)
    if photo is None:
        raise InputError("The selected capture was removed from this specimen.")
    if identifier == plane["photo_id"]:
        corners = plane["corners"]
    else:
        registration = tile["registrations"].get(identifier)
        if registration is None or registration["base_photo_id"] != plane["photo_id"]:
            raise InputError("Mark corresponding corners on this capture before comparing illumination.")
        corners = registration["corners"]
    transform = homography(DEFAULT_CORNERS, corners)
    destination = tuple(plane["size"])
    rgb, alpha = open_working(blobs, photo)
    linear, valid, clipped = warp_capture(rgb, alpha, destination, transform)
    if np.count_nonzero(valid) < 64:
        raise InputError("The marked specimen plane contains too few usable pixels.")
    return SampleFrame(linear, valid, photo, tuple(transform.ravel()), physical_scale(tile, *destination), clipped)


def active_region(tile, region_id):
    if not region_id:
        return None
    found = next((region for region in tile["regions"] if region["id"] == region_id), None)
    if not found:
        raise InputError("The chosen sampling region has been removed.")
    if found["purpose"] == "ignore":
        raise InputError("An exclusion region cannot be used as the inspection area.")
    return found


def polygon_mask(extent, vertices):
    width, height = extent
    grid = Image.new("1", extent)
    points = [(float(x * width - 0.5), float(y * height - 0.5)) for x, y in vertices]
    ImageDraw.Draw(grid).polygon(points, fill=1)
    return np.asarray(grid, dtype=bool)


def exclusion_mask(extent, regions):
    width, height = extent
    mask = np.zeros((height, width), dtype=bool)
    for region in regions:
        if region["purpose"] == "ignore":
            mask |= polygon_mask(extent, region["vertices"])
    return mask


def region_domain(extent, tile, selected=None):
    width, height = extent
    area = np.ones((height, width), dtype=bool) if selected is None else polygon_mask(extent, selected["vertices"])
    return area & ~exclusion_mask(extent, tile["regions"])


def analysis_mask(frame, area, *, minimum=36, allow_clipped=True):
    support = area & frame.valid
    if not allow_clipped:
        support &= ~frame.clipped
    if np.count_nonzero(support) < minimum:
        raise InputError("The selection contains too few supported pixels.")
    return support


def normalized_bbox(bounds, extent):
    width, height = extent
    row, col = bounds
    return [col.start / width, row.start / height, col.stop / width, row.stop / height]


def extent_crop(mask):
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    return int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)


def line_grid(extent, start, end, count, band_width):
    width, height = extent
    origin = np.array(start) * [width, height] - 0.5
    finish = np.array(end) * [width, height] - 0.5
    delta = finish - origin
    length = float(np.linalg.norm(delta))
    if length < 3:
        raise InputError("Mark two line endpoints more than three pixels apart.")
    perpendicular = np.array([-delta[1], delta[0]]) / length
    steps = max(int(math.ceil(band_width)), 1)
    transverse = np.linspace(-band_width / 2, band_width / 2, steps) if steps > 1 else np.array([0])
    center = origin[None, None, :] + np.linspace(0, 1, count)[:, None, None] * delta[None, None, :]
    coords = center + transverse[None, :, None] * perpendicular[None, None, :]
    return (coords[..., 1], coords[..., 0]), delta


def measure_line_samples(image, mask, coords):
    weight = ndimage.map_coordinates(mask.astype(np.float32), coords, order=1, mode="constant", cval=0)
    supported = weight >= 0.999
    valid_count = supported.sum(axis=1)
    if not valid_count.any():
        raise InputError("The line never crosses supported specimen pixels.")
    channels = image.shape[2] if image.ndim > 2 else 1
    stack = image if image.ndim > 2 else image[:, :, None]
    aggregate = np.zeros((len(valid_count), channels), dtype=np.float32)
    for index in range(channels):
        channel = ndimage.map_coordinates(stack[:, :, index], coords, order=1, mode="constant", cval=0)
        mean = np.sum(np.where(supported, channel, 0), axis=1)
        np.divide(mean, valid_count, out=aggregate[:, index], where=valid_count > 0)
    return aggregate, valid_count
