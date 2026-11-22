"""Estimate isolated yarn-knot footprints on a registered fabric."""
import numpy as np
from PIL import Image
from scipy import ndimage

from .art import base_rgba
from .model import InputError


def local_ink_contrast(linear, support, radius):
    # Grey closing estimates the light surface across narrow dark marks. Mask the
    # support erosion too: a missing margin cannot supply a white reference.
    luminance = linear @ np.array([.2126, .7152, .0722], dtype=np.float32)
    footprint = (radius * 2 + 1, radius * 2 + 1)
    paper = ndimage.grey_closing(luminance, size=footprint, mode="nearest")
    valid = ndimage.minimum_filter(support, size=footprint, mode="constant", cval=False)
    return np.clip((paper - luminance) / np.maximum(paper, .025), 0, 1), valid


def component_reading(points, extent, scale):
    y, x = points
    xx = x.astype(float) + .5
    yy = y.astype(float) + .5
    center = np.array([xx.mean(), yy.mean()])
    offsets = np.column_stack((xx - center[0], yy - center[1]))
    if scale:
        offsets *= np.asarray(scale)
    eigenvalues, vectors = np.linalg.eigh(offsets.T @ offsets / len(x))
    minor, major = np.maximum(eigenvalues, 0)
    ratio = np.sqrt(major / minor) if minor > 0 else None
    axis = vectors[:, 1]
    angle = np.degrees(np.arctan2(axis[1], axis[0])) % 180
    return {
        "area_px": int(len(x)),
        "area_mm2": round(float(len(x) * scale[0] * scale[1]), 5) if scale else None,
        "axis_ratio": round(float(ratio), 3) if ratio is not None else None,
        "major_angle_deg": round(float(angle), 2) if ratio and ratio > 1.08 else None,
        "center": np.round(center / np.asarray(extent), 6).tolist(),
    }


def measure_dots(frame, area, linear, options):
    if options["minimum_area"] >= options["maximum_area"]:
        raise InputError("The smallest dot must be smaller than the largest dot.")
    support = area & frame.valid & ~frame.clipped
    contrast, valid = local_ink_contrast(linear, support, options["radius"])
    if np.count_nonzero(valid) < 144:
        raise InputError("The region has too little supported fabric for this background radius.")
    ink = valid & (contrast >= options["contrast"])
    labels, count = ndimage.label(ink, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count+1)
    border = valid & ~ndimage.binary_erosion(valid, structure=np.ones((3, 3), dtype=bool))
    touches = set(np.unique(labels[border]))
    selected = np.zeros(count+1, dtype=bool)
    objects = ndimage.find_objects(labels)
    rows = []
    for index, bounds in enumerate(objects, start=1):
        if (bounds is None or index in touches or
                not options["minimum_area"] <= sizes[index] <= options["maximum_area"]):
            continue
        y, x = np.nonzero(labels[bounds] == index)
        y += bounds[0].start
        x += bounds[1].start
        selected[index] = True
        rows.append({"dot": len(rows)+1, **component_reading((y, x), frame.extent, frame.scale)})
    accepted = selected[labels]
    uncertain = ink & ~accepted
    image = base_rgba(frame)
    picture = np.asarray(image).copy()
    outline = accepted & ~ndimage.binary_erosion(accepted)
    picture[outline, :3] = [83, 230, 207]
    picture[uncertain, :3] = (.65*picture[uncertain, :3] + .35*np.array([253, 166, 63])).astype(np.uint8)
    image = Image.fromarray(picture)
    areas = [row["area_px"] for row in rows]
    denominator = np.count_nonzero(valid)
    return {
        "dot_count": len(rows),
        "median_area_px": round(float(np.median(areas)), 2) if rows else None,
        "ink_area_percent": round(100 * int(accepted.sum()) / denominator, 3),
        "unresolved_ink_percent": round(100 * int(uncertain.sum()) / denominator, 3),
        "supported_pixels": int(denominator),
        "rows": rows[:600], "table_truncated": len(rows) > 600,
        "note": "Teal outlines mark complete, size-qualified knot components. Amber candidates is oversized, small or cut by the analysis boundary. Nearly round knots have no reported axis direction. Table shows the first 600 knots in reading order.",
    }, image
