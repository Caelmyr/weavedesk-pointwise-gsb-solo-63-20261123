"""Read the extent of a single binding strip along a marked specimen axis."""

from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .art import base_rgba
from .color import linear_lab
from .coverage import reference_region, sample_chip
from .geometry import region_domain
from .model import InputError


@dataclass(frozen=True)
class Axis:
    start: np.ndarray
    finish: np.ndarray
    scale: np.ndarray
    direction: np.ndarray
    length: float

    @classmethod
    def from_settings(cls, frame, settings):
        size = np.asarray(frame.extent, dtype=float) - 1
        start = np.asarray(settings["start"]) * size
        finish = np.asarray(settings["end"]) * size
        scale = np.asarray(frame.scale or (1., 1.))
        delta = (finish - start) * scale
        length = float(np.linalg.norm(delta))
        if length < 3 * float(scale.min()):
            raise InputError("Mark distinct points along the sample.")
        return cls(start, finish, scale, delta / length, length)

    @property
    def across(self):
        return np.array([-self.direction[1], self.direction[0]]) / self.scale

    def centers(self, stations):
        return self.start + np.linspace(0., 1., stations)[:, None] * (self.finish - self.start)


def projection(tile, frame, linear, settings):
    lab = linear_lab(linear)
    reference_domain = region_domain(frame.extent, tile)
    paper_region = reference_region(tile, "bare", settings["bare_region"])
    leaf_region = reference_region(tile, "glazed", settings["glazed_region"])
    paper, _, _ = sample_chip(frame, lab, reference_domain, paper_region)
    leaf, _, _ = sample_chip(frame, lab, reference_domain, leaf_region)
    delta = leaf - paper
    distance = float(np.linalg.norm(delta))
    if distance < 5:
        raise InputError("The mount and binding samples are too similar for a width reading.")
    shifted = lab - paper
    fraction = shifted @ delta / (distance ** 2)
    residual = shifted - fraction[..., None] * delta
    acceptable = np.linalg.norm(residual, axis=2) < distance * .46
    return fraction, acceptable


def measure_crossing(trace, trusted, coords):
    middle = len(trace) // 2
    if not trusted[middle]:
        return "obscured axis", None, None
    if trace[middle] < .52:
        return "no binding at axis", None, None
    left = middle
    while left > 0 and trusted[left - 1] and trace[left - 1] >= .52:
        left -= 1
    right = middle
    while right + 1 < len(trace) and trusted[right + 1] and trace[right + 1] >= .52:
        right += 1
    if left == 0 or right == len(trace) - 1:
        return "outside sampled band", None, None
    if not trusted[left - 1] or not trusted[right + 1]:
        return "edge obscured", None, None
    return "observed", coords[left], coords[right]


def blade_width(tile, frame, area, linear, settings):
    axis = Axis.from_settings(frame, settings)
    fraction, on_axis = projection(tile, frame, linear, settings)
    support = frame.valid & area & ~frame.clipped & on_axis
    centers = axis.centers(settings["stations"])
    # The radius is a fraction of the smaller photographed extent in physical space.
    half = float(min(np.asarray(frame.extent) * axis.scale)) * settings["half_band"]
    count = min(450, max(2, int(np.ceil(half / float(axis.scale.min())))))
    offsets = np.linspace(-half, half, count * 2 + 1)
    points = centers[:, None, :] + offsets[None, :, None] * axis.across
    coords = (points[..., 1], points[..., 0])
    values = ndimage.map_coordinates(fraction, coords, order=1, mode="constant", cval=0)
    trusted = ndimage.map_coordinates(support.astype(np.float32), coords, order=1, mode="constant", cval=0) >= .9999
    image = base_rgba(frame)
    draw = ImageDraw.Draw(image)
    draw.line([tuple(axis.start), tuple(axis.finish)], fill=(252, 222, 107), width=2)
    rows = []
    for idx, center in enumerate(centers):
        condition, first, last = measure_crossing(values[idx], trusted[idx], points[idx])
        width = float(np.linalg.norm((last - first) * axis.scale)) if first is not None else None
        extent = np.asarray(frame.extent, dtype=float)
        ends = [np.round(point / extent, 6).tolist() for point in (first, last)] if first is not None else None
        rows.append({
            "position": round(idx * axis.length / (len(centers) - 1), 3),
            "center": np.round(center / extent, 6).tolist(),
            "state": condition,
            "width": round(width, 3) if width is not None else None,
            "endpoints": ends,
        })
        if first is not None:
            draw.line([tuple(first), tuple(last)], fill=(159, 222, 187), width=1)
            for px, py in (first, last):
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(222, 244, 215))
        else:
            px, py = center
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(245, 139, 93))
    widths = [row["width"] for row in rows if row["width"] is not None]
    metrics = {
        "axis_length": round(axis.length, 3),
        "length_unit": "mm" if frame.scale else "px",
        "observed_stations": len(widths),
        "requested_stations": len(rows),
        "maximum_width": round(max(widths), 3) if widths else None,
        "stations": rows,
        "note": "Widths follow cross-sections of the visible binding attached to the marked axis. An obscured edge has no width reading.",
    }
    return metrics, image
