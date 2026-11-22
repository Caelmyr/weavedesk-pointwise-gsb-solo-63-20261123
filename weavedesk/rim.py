from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from .color import delta_e, float_list, linear_lab, quantile_rows
from .model import InputError
from .photos import uint8_image


@dataclass(frozen=True)
class EdgeSide:
    label: str
    index: int
    axis: int
    flip: bool


SIDES = (
    EdgeSide("top", 0, 0, False),
    EdgeSide("right", 1, 1, True),
    EdgeSide("bottom", 2, 0, True),
    EdgeSide("left", 3, 1, False),
)
EDGE_COLORS = np.array([[237, 180, 130], [133, 199, 179], [184, 180, 221], [227, 211, 127]], dtype=np.float32) / 255


def sided_distance(shape, edges):
    height, width = shape
    yy, xx = np.indices((height, width))
    distances = np.stack((yy, width - 1 - xx, height - 1 - yy, xx), axis=2).astype(np.float32)
    for side, width_edge in zip(SIDES, edges):
        distances[:, :, side.index] /= max(width_edge, 1)
    nearest = np.argmin(distances, axis=2)
    min_distance = np.take_along_axis(distances, nearest[:, :, None], axis=2)[:, :, 0]
    return nearest, min_distance


def physical_margin(frame, settings):
    width, height = frame.extent
    selected = settings["band"]
    if settings["unit"]=="mm":
        if frame.scale is None:
            raise InputError("Add the specimen dimensions before using a millimeter rim band.")
        levels = [selected / frame.scale[1], selected / frame.scale[0]]
    else:
        levels = [height * selected / 100, width * selected / 100]
    if min(levels) < 2.0:
        raise InputError("Set a rim band at least two registered pixels deep.")
    if levels[0] >= height * 0.25 or levels[1] >= width * 0.25:
        raise InputError("Rim thickness must leave enough room for interior comparison.")
    return [levels[0], levels[1], levels[0], levels[1]]


def distance_bands(frame, settings):
    bands = physical_margin(frame, settings)
    nearest, relative = sided_distance(frame.valid.shape, bands)
    cutoff = settings["interior_multiple"]
    return nearest, relative, bands, relative >= cutoff


def rim_swath_stats(lab, aim, reference, scale):
    count = int(np.count_nonzero(aim))
    if count < 30:
        return None
    data = lab[aim]
    median = np.median(data, axis=0)
    spread = delta_e(data, reference)
    chroma = np.linalg.norm(data[:, 1:], axis=1)
    return {
        "pixels": count,
        "median_lab": float_list(median, 5),
        "median_from_center_delta_e": round(float(delta_e(median, reference)), 4),
        "pixel_delta_quantiles": quantile_rows(spread, (0.10, 0.50, 0.90, 0.95)),
        "lightness_quantiles": quantile_rows(data[:, 0], (0.10, 0.50, 0.90)),
        "chroma_quantiles": quantile_rows(chroma, (0.10, 0.50, 0.90)),
        "supported_area_mm2": round(scale[0] * scale[1] * count, 5) if scale else None,
    }


def relative_swath_rows(lab, selector, distance, reference, frame, divisions):
    positions = np.linspace(0, 1, divisions + 1)
    rows = []
    for lo, hi in zip(positions, positions[1:]):
        strip = selector & (distance >= lo) & (distance < hi)
        data = rim_swath_stats(lab, strip, reference, frame.scale)
        if data is None:
            continue
        rows.append({
            "fraction_from_edge": [round(float(lo), 4), round(float(hi), 4)],
            **data,
        })
    return rows


def bands_for_side(frame, side, lab, available, area, nearest, distance, median, divisions):
    requested = (nearest == side.index) & (distance < 1) & area
    native = available & requested
    band_count = int(np.count_nonzero(requested))
    result = {
        "side": side.label,
        "requested_pixels": band_count,
        "stats": rim_swath_stats(lab, native, median, frame.scale),
        "swaths": relative_swath_rows(lab, native, distance, median, frame, divisions),
    }
    result["excluded_percent"] = round(100 * (1 - np.count_nonzero(native) / band_count), 3) if band_count else None
    return result


def rim_change_hotspots(labs, reference, rim, frame, threshold):
    delta = delta_e(labs, reference)
    departures = rim & (delta > threshold)
    labels, count = ndimage.label(departures)
    if not count:
        return []
    sizes = np.bincount(labels.ravel())[1:]
    top = np.argsort(sizes, kind="stable")[::-1][:24]
    parts = ndimage.find_objects(labels)
    width, height = frame.extent
    kept = []
    for index in top:
        if sizes[index] < 12:
            continue
        yy, xx = parts[index]
        kept.append({
            "bounds": [xx.start / width, yy.start / height, xx.stop / width, yy.stop / height],
            "pixels": int(sizes[index]),
            "diam_px": round(2 * math.sqrt(sizes[index] / np.pi), 2),
        })
    return kept


def rim_overlay(frame, control, nearby, distance, nearest, hotspots):
    display = frame.display() * 0.52
    display[control] = 0.45 * display[control] + 0.55 * np.array([0.40, 0.56, 0.53], dtype=np.float32)
    for side in SIDES:
        known = nearby & (nearest == side.index)
        depth = np.clip(distance[known], 0, 1)[:, None]
        display[known] = display[known] * (0.4 + depth * 0.28) + EDGE_COLORS[side.index] * (0.60 - depth * 0.28)
    picture = uint8_image(display, frame.valid.astype(np.float32))
    if hotspots:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(picture)
        width, height = frame.extent
        for part in hotspots:
            x0, y0, x1, y1 = part["bounds"]
            draw.rectangle((x0 * width, y0 * height, x1 * width, y1 * height), outline=(255, 235, 178), width=2)
    return picture


def measure_rim(frame, linear, area, options):
    labs = linear_lab(linear)
    available = area & frame.valid & ~frame.clipped
    nearest, distance, pixels, core = distance_bands(frame, options)
    reference = core & available
    if np.count_nonzero(reference) < 40:
        raise InputError("The inspection selection must include an unclipped interior area.")
    median = np.median(labs[reference], axis=0)
    rim = (distance < 1) & available
    if np.count_nonzero(rim) < 45:
        raise InputError("The inspection selection contains too little of the specimen rim.")
    summaries = [
        bands_for_side(
            frame, side, labs, available, area, nearest, distance, median, options["depth_bins"],
        )
        for side in SIDES
    ]
    center = rim_swath_stats(labs, reference, median, frame.scale)
    delta_stats = rim_swath_stats(labs, rim, median, frame.scale)
    hotspots = rim_change_hotspots(labs, median, rim, frame, options["highlight_delta"])
    result = {
        "rim_band_pixels": float_list(pixels, 3),
        "interior": center,
        "all_edges": delta_stats,
        "sides": summaries,
        "large_color_departures": hotspots,
        "note": "Compare image color near the marked specimen edges against the interior. The result depends on the tile plane.",
    }
    return result, rim_overlay(frame, reference, rim, distance, nearest, hotspots)
