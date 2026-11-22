"""Plan a lining patch for a broken band and meter a repair seam in millimetres.

Each photographed piece is registered on its own millimetre plane. A plan
reads the two yarn directions of the broken band and of each candidate lining,
fits one continuous 28 x 18 mm patch into cloth that is actually photographed,
and states where that patch lands under the opening, which way the cloth's own
marked upper end has to face, how the old blue edging still lines up, and what
the chosen piece leaves uncovered or unread.

A seam trace then counts only the millimetres where the same point is visible
as old band, lies on the placed cut, and is visible as cloth in the lining
photograph. Stretches over the opening, off the patch, or outside either
photograph are listed separately and never added to the seam length.
"""

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .analysis import capture_provenance, specimen_provenance
from .art import INK, SHADOW, base_rgba, font, stored_preview
from .color import float_list
from .defects import gaussian_supported
from .geometry import polygon_mask, specimen_frame
from .model import InputError, collection, dictionary, identity, numeric_pair, scalar, string
from .photos import luminance
from .weave import spectrum


PATCH_LENGTH_MM = 28.0
PATCH_WIDTH_MM = 18.0
GRAIN_PITCH_MIN_MM = 0.3
GRAIN_PITCH_MAX_MM = 3.0
MIN_GRAIN_SAMPLES = 520
MIN_GRAIN_RATIO = 5.0
GRAIN_PEAK_SEPARATION = 25.0
MIN_OPENING_PIXELS = 24
SEAM_STEP_MM = 0.05
BLUE_MIN_DOMINANCE = 8e-6
BLUE_CHROMA = 0.09
BLUE_MIN_PIXELS = 24
TIGHT_MARGIN_MM = 2.0
GRAIN_AGREEMENT_DEGREES = 10.0
UP_PREFERENCE_DEGREES = 45.0

COUNTED = "old band and lining both bear"
GAP = "crosses opening"
OFF_PATCH = "off patch"
UNSEEN_BAND = "not visible on band photo"
UNCLEAR_BAND = "not clear on band photo"
NOT_BAND_CLOTH = "not old band cloth"
UNSEEN_CLOTH = "not clear on lining photo"
NOT_LINING_CLOTH = "not lining cloth"
OUTSIDE_CLOTH = "outside lining photo"
STATUS_ORDER = (
    COUNTED, GAP, OFF_PATCH, UNSEEN_BAND, UNCLEAR_BAND, NOT_BAND_CLOTH,
    UNSEEN_CLOTH, NOT_LINING_CLOTH, OUTSIDE_CLOTH,
)

STATUS_COLORS = {
    COUNTED: (103, 224, 168),
    GAP: (239, 116, 102),
    OFF_PATCH: (236, 176, 92),
    UNSEEN_BAND: (148, 165, 184),
    UNCLEAR_BAND: (196, 186, 168),
    NOT_BAND_CLOTH: (176, 158, 128),
    UNSEEN_CLOTH: (148, 165, 184),
    NOT_LINING_CLOTH: (176, 158, 128),
    OUTSIDE_CLOTH: (148, 165, 184),
}
PATCH_COLOR = (116, 214, 178)
OPENING_COLOR = (239, 116, 102)
BLUE_COLOR = (104, 152, 226)
CLOTH_COLOR = (255, 243, 213)


def wrap180(angle):
    return (angle + 90.0) % 180.0 - 90.0


def angle_difference(first, second):
    return abs(wrap180(first - second))


def unit(angle_degrees):
    radians = math.radians(angle_degrees)
    return np.array([math.cos(radians), math.sin(radians)], dtype=np.float64)


def rotate(vector, degrees):
    """Turn an image-plane vector; positive degrees advance its atan2 heading."""
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.array([cosine * vector[0] - sine * vector[1], sine * vector[0] + cosine * vector[1]], dtype=np.float64)


def compass(angle_degrees):
    radians = math.radians(angle_degrees)
    east, south = math.cos(radians), math.sin(radians)
    vertical = "down" if south > 0.38 else "up" if south < -0.38 else ""
    horizontal = "right" if east > 0.38 else "left" if east < -0.38 else ""
    if vertical and horizontal:
        return f"toward the lower {horizontal}" if vertical == "down" else f"toward the upper {horizontal}"
    if vertical:
        return "straight down" if vertical == "down" else "straight up"
    if horizontal:
        return f"to the {horizontal}"
    return "across the band"


def extent_mm(frame):
    if not frame.scale:
        raise InputError("Give this photograph a width and height in millimetres before planning a patch.")
    width, height = frame.extent
    return np.array([width * frame.scale[0], height * frame.scale[1]], dtype=np.float64)


def photo_support(frame):
    return frame.valid & ~frame.clipped


def cloth_states(frame, pitch_mm, region=None, exclude=None):
    """Split photographed support into cloth, weak/defocused texture and smooth furniture."""
    support = photo_support(frame)
    if region is not None:
        support = support & polygon_mask(frame.extent, region["vertices"])
    if exclude is not None:
        support = support & ~exclude
    if not support.any():
        raise InputError("This photograph has no supported cloth pixels.")
    gray = luminance(frame.linear)
    pitch_px = max(1.0, pitch_mm / float(min(frame.scale)))
    sigma = max(1.2, pitch_px * 0.5)
    mean = ndimage.gaussian_filter(np.where(support, gray, 0), sigma, mode="constant", cval=0)
    second = ndimage.gaussian_filter(np.where(support, gray * gray, 0), sigma, mode="constant", cval=0)
    spread = np.sqrt(np.maximum(second - mean * mean, 0))
    detail = spread / np.maximum(mean, 1e-4)
    level = float(np.median(detail[support]))
    clear = support & (detail > max(level * 0.42, 0.012))
    structure = ndimage.generate_binary_structure(2, 2)
    clear = ndimage.binary_closing(clear, structure, iterations=2)
    clear = ndimage.binary_opening(clear, structure, iterations=max(1, int(round(pitch_px * 0.4))))
    labels, count = ndimage.label(clear, structure)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        # A torn band arrives in more than one piece; keep every substantial piece.
        keep = sizes >= max(sizes.max() * 0.12, 40)
        keep[0] = False
        clear = keep[labels]
    ratio = np.count_nonzero(clear) / max(int(np.count_nonzero(support)), 1)
    detected = 0.2 <= ratio <= 0.98
    if not detected:
        clear = support
    weak = support & ~clear & (detail > max(level * 0.18, 0.005))
    smooth = support & ~clear & ~weak
    structure = ndimage.generate_binary_structure(2, 2)
    weak = ndimage.binary_closing(weak, structure, iterations=1)
    return {"cloth": clear, "unclear": weak, "smooth": smooth, "detected": bool(detected), "level": level}


def cloth_mask(frame, pitch_mm, region=None, exclude=None):
    states = cloth_states(frame, pitch_mm, region, exclude)
    return states["cloth"], states["detected"]


def grain_axes(frame, support):
    """Return two yarn line directions in millimetres with their spacing."""
    if not frame.scale:
        raise InputError("Give the fabric's photographed size in millimetres before reading grain.")
    pitch_px = 1.0 / float(min(frame.scale))
    core = ndimage.binary_erosion(support, iterations=max(1, int(round(2.0 * pitch_px))))
    if np.count_nonzero(core) < MIN_GRAIN_SAMPLES:
        core = support
    if np.count_nonzero(core) < MIN_GRAIN_SAMPLES:
        raise InputError("The visible cloth is too small to resolve two yarn directions.")
    rows, cols = np.nonzero(core)
    box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    mask = core[box]
    gray = luminance(np.log(np.maximum(frame.linear, 1e-4)))[box]
    # Remove only the slow illumination drift. A wider blur would smooth the yarn
    # repeat itself and leave the photograph's edge as the strongest feature.
    background, _ = gaussian_supported(gray, np.ones_like(mask, dtype=bool), max(1.0, 0.33 * pitch_px))
    power, frequencies, normals = spectrum(gray - background, _taper(mask, frame), frame.scale)
    band = (frequencies >= 1 / GRAIN_PITCH_MAX_MM) & (frequencies <= 1 / GRAIN_PITCH_MIN_MM)
    band[0] = False
    if np.count_nonzero(band) < 24:
        raise InputError("This patch is too small to resolve a yarn spacing in the 0.3 to 3 mm range.")
    bins = 720
    seats = np.rint(normals * (bins / 180.0)).astype(int) % bins
    weight = np.bincount(seats[band], weights=power[band], minlength=bins)
    smoothed = ndimage.gaussian_filter1d(weight, 2.0, mode="wrap")
    picked = []
    for candidate in np.argsort(smoothed)[::-1]:
        seat = float(candidate * 180.0 / bins)
        if all(angle_difference(seat, item) >= GRAIN_PEAK_SEPARATION for item in picked):
            picked.append(seat)
        if len(picked) == 2:
            break
    if len(picked) < 2:
        raise InputError("Only one yarn direction is visible here. Show a patch with both warp and weft.")
    readings = []
    for seat in sorted(picked):
        sector = band & (np.abs(((normals - seat + 90) % 180) - 90) <= 3.0)
        if not sector.any():
            raise InputError("A yarn direction is too weak to measure on this photograph.")
        rows, cols = np.nonzero(sector)
        values = power[rows, cols]
        peak = float(values.max())
        floor = float(np.median(values))
        if peak / max(floor, 1e-12) < MIN_GRAIN_RATIO:
            raise InputError("A yarn direction is not clear enough to trust on this photograph.")
        sector_normals = (normals[rows, cols] + 90.0) % 180.0
        middle = float(np.median(sector_normals))
        aligned = np.abs(((sector_normals - middle + 90) % 180) - 90) <= 3.0
        if aligned.any():
            rows, cols, values = rows[aligned], cols[aligned], values[aligned]
        ranked = np.argsort(values)[::-1]
        strong = ranked[values[ranked] >= 0.10 * peak]
        if not len(strong):
            strong = ranked[:1]
        # A woven repeat also shows half- and third-pitch harmonics. Group peaks
        # with similar spacing and take the longest group that still carries
        # real weight as the yarn's own repeat.
        pitches = 1.0 / frequencies[rows[strong], cols[strong]]
        weights = values[strong] / peak
        order = np.argsort(pitches)
        groups = []
        for index in order:
            if groups and abs(pitches[index] - groups[-1][-1]) <= 0.12 * max(pitches[index], 1e-9):
                groups[-1].append(pitches[index])
            else:
                groups.append([pitches[index]])
        scored = [
            (float(np.median(group)), float(np.sum(weights[np.isin(pitches, group)])))
            for group in groups
        ]
        viable = [item for item in scored if item[1] >= 0.25 * float(np.max(weights))]
        pitch = max(viable or scored, key=lambda item: item[0])[0] if scored else 1.0 / float(frequencies[rows[ranked[0]], cols[ranked[0]]])
        amid = (normals[rows, cols] + 90.0) % 180.0
        family = np.abs(frequencies[rows, cols] - 1.0 / pitch) <= 0.35 / pitch
        line_angle = float(np.median(amid[family])) if family.any() else middle
        readings.append({
            "angle": round(line_angle % 180.0, 2),
            "pitch_mm": round(pitch, 3),
            "peak_to_background": round(peak / max(floor, 1e-12), 1),
        })
    readings.sort(key=lambda item: item["angle"])
    crossing = angle_difference(readings[0]["angle"], readings[1]["angle"])
    if crossing < 35.0:
        raise InputError("The two measured yarn directions lie too close together to place a patch.")
    return readings, round(crossing, 2)


def _taper(mask, frame):
    """Smooth the analysis weight down at the cloth boundary so the cut edge is not a spectral line."""
    pitch = 1.0 / float(min(frame.scale))
    depth = ndimage.distance_transform_edt(mask)
    window = np.clip(depth / max(1.0, 2.6 * pitch), 0.0, 1.0)
    window = ndimage.gaussian_filter(window, max(0.8, 0.4 * pitch), mode="constant", cval=0)
    return (window * mask).astype(np.float32)


def opening_metrics(frame, region):
    stencil = polygon_mask(frame.extent, region["vertices"])
    total = int(np.count_nonzero(stencil))
    if total < MIN_OPENING_PIXELS:
        raise InputError("Mark a larger opening region before planning a lining patch.")
    scale = np.asarray(frame.scale)
    samples = np.asarray(region["vertices"], dtype=np.float64) * extent_mm(frame)
    rows, cols = np.nonzero(stencil)
    return {
        "region_id": region["id"],
        "region_name": region["name"],
        "vertices": region["vertices"],
        "raster_pixels": total,
        "area_mm2": round(total * scale[0] * scale[1], 3),
        "unsupported_percent": round(100.0 * np.count_nonzero(stencil & ~frame.valid) / total, 2),
        "bounds_mm": [round(float(value), 3) for value in np.r_[samples.min(axis=0), samples.max(axis=0)]],
        "axis_width_mm": round(float(np.ptp(samples[:, 0])), 3),
        "axis_height_mm": round(float(np.ptp(samples[:, 1])), 3),
        "centroid_mm": [
            round(float(np.mean(cols + 0.5) * scale[0]), 3),
            round(float(np.mean(rows + 0.5) * scale[1]), 3),
        ],
        "mask": stencil,
        "mm": samples,
    }


def long_direction(frame, cloth, grain, blue):
    rows, cols = np.nonzero(cloth)
    if len(rows) < MIN_GRAIN_SAMPLES:
        raise InputError("The visible band is too small to choose its long direction.")
    points = np.stack((cols + 0.5, rows + 0.5), axis=1) * np.asarray(frame.scale)
    spans = [float(np.ptp(points @ unit(item["angle"]))) for item in grain]
    index = int(np.argmax(spans))
    angles = [item["angle"] for item in grain]
    note = "The long direction follows the larger visible extent of the band."
    if blue and blue.get("line_angle") is not None:
        blue_angle = blue["line_angle"]
        index = int(np.argmin([angle_difference(angle, blue_angle) for angle in angles]))
        note = "The long direction follows the blue edging, which should run along the band."
    return {
        "angle": angles[index],
        "other_angle": angles[1 - index],
        "long_extent_mm": round(spans[index], 3),
        "short_extent_mm": round(spans[1 - index], 3),
        "note": note,
    }


def split_two(values):
    """One-dimensional two-means split for the two blue edge lines."""
    ordered = np.sort(values)
    if len(ordered) < 4 or float(np.ptp(ordered)) < 1.2:
        return None
    low, high = float(np.quantile(ordered, 0.1)), float(np.quantile(ordered, 0.9))
    for _ in range(30):
        middle = (low + high) / 2
        near = ordered[ordered <= middle]
        far = ordered[ordered > middle]
        if not len(near) or not len(far):
            return None
        low, high = float(np.mean(near)), float(np.mean(far))
    middle = (low + high) / 2
    if abs(high - low) < 1.2:
        return None
    return middle


def blue_edge(frame, opening, long_angle, cloth):
    """Fit the old blue edging either side of the break, edge line by edge line."""
    if not frame.scale:
        return None
    red, green, blue = (frame.linear[:, :, index] for index in range(3))
    level = np.maximum(red + green + blue, 1e-5)
    chroma_blue = (blue - red) / level
    dominant = (
        (chroma_blue > BLUE_CHROMA)
        & ((blue - green) / level > BLUE_CHROMA * 0.45)
        & photo_support(frame)
    )
    if np.count_nonzero(dominant) < BLUE_MIN_PIXELS:
        return None
    rows, cols = np.nonzero(dominant)
    points = np.stack((cols + 0.5, rows + 0.5), axis=1) * np.asarray(frame.scale)
    along = points @ unit(long_angle)
    across = points @ unit(long_angle + 90.0)
    opening_along = float(np.asarray(opening["centroid_mm"]) @ unit(long_angle))
    split = split_two(across)
    groups = [("near edge", across <= split), ("far edge", across > split)] if split is not None else [("edge line", np.ones(len(across), dtype=bool))]

    def fit(selector):
        part = points[selector]
        if len(part) < max(16, BLUE_MIN_PIXELS // 2):
            return None
        centroid = np.mean(part, axis=0)
        eigen, vectors = np.linalg.eigh(np.cov((part - centroid).T))
        direction = vectors[:, -1]
        angle = math.degrees(math.atan2(direction[1], direction[0])) % 180
        length = float(np.sqrt(max(eigen[-1], 0.0)) * 3.4)
        thickness = float(np.sqrt(max(eigen[0], 0.0)) * 3.4)
        if length < 1.5 or length < thickness * 1.3:
            return None
        toward = unit(angle)
        if np.dot(toward, unit(long_angle)) < 0:
            toward = -toward
        return {
            "pixels": int(len(part)),
            "angle": round(float(angle), 2),
            "angle_from_long": round(angle_difference(angle, long_angle), 2),
            "length_mm": round(length, 2),
            "thickness_mm": round(thickness, 2),
            "centroid_mm": float_list(centroid, 3),
            "offset_mm": round(float(np.mean(part @ unit(long_angle + 90.0))), 3),
            "along_mm": round(float(np.mean(part @ unit(long_angle))), 3),
            "slope": round(float(np.dot(toward, unit(long_angle + 90.0)) / max(np.dot(toward, unit(long_angle)), 1e-6)), 5),
        }

    edges = []
    for name, selector in groups:
        before = fit(selector & (along < opening_along))
        after = fit(selector & (along >= opening_along))
        edge = {"edge": name, "offset_mm": round(float(np.mean(across[selector])), 3)}
        if before and after:
            # Compare the two fits at the same station along the band, not at their
            # own centroids, which move when one side of the break is longer.
            at_before = before["offset_mm"] + (opening_along - before["along_mm"]) * before["slope"]
            at_after = after["offset_mm"] + (opening_along - after["along_mm"]) * after["slope"]
            edge.update({
                "before": before,
                "after": after,
                "angle_difference": round(angle_difference(before["angle"], after["angle"]), 2),
                "offset_step_mm": round(abs(at_after - at_before), 3),
                "offset_step_note": "Each fit is extended to the opening station before the two are compared.",
            })
        elif before or after:
            edge["before"] = before
            edge["after"] = after
            edge["visible"] = "before the break" if before else "after the break"
        else:
            continue
        edges.append(edge)
    if not edges:
        return None
    longest = max(
        (item for edge in edges for item in (edge.get("before"), edge.get("after")) if item),
        key=lambda item: item["length_mm"],
    )
    total = sum(
        item["length_mm"] for edge in edges for item in (edge.get("before"), edge.get("after")) if item
    )
    return {
        "edges": edges,
        "line_angle": longest["angle"],
        "line_runs_mm": round(total, 2),
        "note": (
            "Each edge line of the blue edging is fitted before and after the break. Where both fits exist, "
            "the offset step and angle difference show what still has to be brought together."
        ),
    }


def rectangle(center_mm, angle, length=PATCH_LENGTH_MM, width=PATCH_WIDTH_MM):
    along = unit(angle)
    across = unit(angle + 90.0)
    center = np.asarray(center_mm, dtype=np.float64)
    return np.asarray([
        center - along * length / 2 - across * width / 2,
        center + along * length / 2 - across * width / 2,
        center + along * length / 2 + across * width / 2,
        center - along * length / 2 + across * width / 2,
    ])


def normalized(points_mm, span_mm):
    return (np.asarray(points_mm, dtype=np.float64) / np.asarray(span_mm)).tolist()


def stencil(frame, points_mm):
    return polygon_mask(frame.extent, normalized(points_mm, extent_mm(frame)))


def placement_geometry(band, opening, long_angle):
    along = opening["mm"] @ unit(long_angle)
    across = opening["mm"] @ unit(long_angle + 90.0)
    center_along = (along.min() + along.max()) / 2
    center_across = (across.min() + across.max()) / 2
    center = unit(long_angle) * center_along + unit(long_angle + 90.0) * center_across
    margins = [
        PATCH_LENGTH_MM / 2 - abs(along.min() - center_along),
        PATCH_LENGTH_MM / 2 - abs(along.max() - center_along),
        PATCH_WIDTH_MM / 2 - abs(across.min() - center_across),
        PATCH_WIDTH_MM / 2 - abs(across.max() - center_across),
    ]
    corners = rectangle(center, long_angle)
    mask = stencil(band, corners)
    covered = int(np.count_nonzero(mask & opening["mask"]))
    total = int(np.count_nonzero(opening["mask"]))
    fraction = covered / total if total else 0.0
    span = extent_mm(band)
    grid = np.linspace(-0.5, 0.5, 41)
    outside = sum(
        bool(np.any(center + unit(long_angle) * (u * PATCH_LENGTH_MM) + unit(long_angle + 90.0) * (v * PATCH_WIDTH_MM) < 0)
             or np.any(center + unit(long_angle) * (u * PATCH_LENGTH_MM) + unit(long_angle + 90.0) * (v * PATCH_WIDTH_MM) > span))
        for u in grid for v in grid
    )
    return {
        "center_mm": float_list(center, 3),
        "corners_mm": float_list(corners, 3),
        "polygon": normalized(corners, span),
        "margin_mm": round(float(min(margins)), 3),
        "opening_extent_mm": [round(float(np.ptp(along)), 3), round(float(np.ptp(across)), 3)],
        "covered_percent": round(100.0 * fraction, 3),
        "uncovered_opening_mm2": round(opening["area_mm2"] * (1.0 - fraction), 3),
        "outside_photo_percent": round(100.0 * outside / grid.size ** 2, 2),
        "_center": center,
        "_corners": corners,
    }


def cut_geometry(lining, cloth, angle):
    span = extent_mm(lining)
    depth = ndimage.distance_transform_edt(cloth) * float(min(lining.scale))
    rows, cols = np.nonzero(cloth)
    if len(rows) < MIN_OPENING_PIXELS:
        raise InputError("No continuous cloth is visible in this lining photograph.")
    order = np.argsort(depth[rows, cols])[::-1]
    stride = max(1, len(order) // 1200)
    best = None
    for index in order[::stride]:
        center = np.array([cols[index] + 0.5, rows[index] + 0.5]) * np.asarray(lining.scale)
        corners = rectangle(center, angle)
        mask = stencil(lining, corners)
        total = int(np.count_nonzero(mask))
        if not total:
            continue
        fraction = np.count_nonzero(mask & cloth) / total
        candidate = (center, float(depth[rows[index], cols[index]]), corners, fraction)
        if best is None or (candidate[3], candidate[1]) > (best[3], best[1]):
            best = candidate
        if fraction >= 0.9999:
            break
    if best is None:
        raise InputError("No cut rectangle could be placed on this lining photograph.")
    center, margin, corners, fraction = best
    mask = stencil(lining, corners)
    visible = np.count_nonzero(mask & photo_support(lining)) / max(int(np.count_nonzero(mask)), 1)
    return {
        "center_mm": float_list(center, 3),
        "corners_mm": float_list(corners, 3),
        "polygon": normalized(corners, span),
        "cloth_margin_mm": round(margin, 3),
        "cloth_support_percent": round(100.0 * float(fraction), 3),
        "photo_support_percent": round(100.0 * float(visible), 3),
        "_center": center,
        "_corners": corners,
    }


def build_candidate(band, lining, opening, band_long, grain_cloth, rotation, cloth_index, up_angle,
                    lining_tile_id, lining_name):
    matched = grain_cloth[cloth_index]["angle"]
    other_cloth = grain_cloth[1 - cloth_index]["angle"]
    long_mismatch = angle_difference(matched + rotation, band_long["angle"])
    short_mismatch = angle_difference(other_cloth + rotation, band_long["other_angle"])
    placement = placement_geometry(band, opening, band_long["angle"])
    cloth, detected = cloth_mask(lining, float(np.mean([item["pitch_mm"] for item in grain_cloth])))
    cut = cut_geometry(lining, cloth, matched)
    mismatch = long_mismatch + short_mismatch
    feasible = (
        placement["margin_mm"] >= 0
        and cut["cloth_support_percent"] >= 99.95
        and cut["photo_support_percent"] >= 99.95
        and mismatch <= GRAIN_AGREEMENT_DEGREES
    )
    reason = None
    if placement["margin_mm"] < 0:
        reason = "The opening is broader than 28 × 18 mm on this placement, so the patch cannot back it."
    elif cut["cloth_support_percent"] < 99.95:
        reason = "No continuous 28 × 18 mm cut lies wholly on visible cloth in this photograph."
    elif cut["photo_support_percent"] < 99.95:
        reason = "Part of the cut falls outside what this photograph shows."
    elif mismatch > GRAIN_AGREEMENT_DEGREES:
        reason = (
            f"This piece's yarn sits {round(mismatch, 1)}° from the old band even at the best turn; "
            "the weave would run visibly across the patch."
        )
    # Where the marked up end of the piece points once the cut is rotated onto
    # the band.
    up = rotate(unit(up_angle), rotation)
    up_degrees = math.degrees(math.atan2(up[1], up[0])) % 360
    upward = abs((up_degrees - 270.0 + 180.0) % 360.0 - 180.0)
    return {
        "lining_tile_id": lining_tile_id,
        "lining_name": lining_name,
        "lining_photo_id": lining.photo["id"],
        "rotation_clockwise_degrees": round(rotation, 2),
        "long_band_angle": round(band_long["angle"], 2),
        "other_band_angle": round(band_long["other_angle"], 2),
        "long_lining_angle": matched,
        "other_lining_angle": other_cloth,
        "grain_mismatch_long_degrees": round(long_mismatch, 2),
        "grain_mismatch_short_degrees": round(short_mismatch, 2),
        "grain_mismatch_total_degrees": round(long_mismatch + short_mismatch, 2),
        "up": {"angle": round(up_degrees, 2), "phrase": compass(up_degrees)},
        "feasible": bool(feasible),
        "reason": reason,
        "cloth_mask_detected": bool(detected),
        "placement": placement,
        "cut": cut,
        "_rotation": rotation,
    }


def public(candidate):
    result = {}
    for key, value in candidate.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            value = {inner: item for inner, item in value.items() if not inner.startswith("_")}
            value = {inner: float_list(item, 3) if isinstance(item, np.ndarray) else item
                     for inner, item in value.items()}
        elif isinstance(value, np.ndarray):
            value = float_list(value, 3)
        result[key] = value
    return result


def choose(records):
    usable = [item for item in records if item.get("placement") and item.get("cut")]
    feasible = [item for item in usable if item["feasible"]]
    pool = feasible or usable or records

    def key(item):
        # Prefer a piece whose yarn actually matches, then the smaller turn of
        # the cloth, then the placement with more overhang around the opening.
        return (
            0 if item["feasible"] else 1,
            round(item["grain_mismatch_total_degrees"], 1),
            abs(item["rotation_clockwise_degrees"]),
            -item["placement"]["margin_mm"],
            -item["cut"]["cloth_margin_mm"],
        )

    return sorted(pool, key=key)[0]


def caveats(chosen, opening, blue):
    placement = chosen["placement"]
    cut = chosen["cut"]
    notes = []
    if not chosen["feasible"]:
        notes.append(chosen["reason"] or "No 28 × 18 mm piece in this photograph can back the opening.")
    if placement["covered_percent"] < 100.0:
        notes.append(f"{placement['uncovered_opening_mm2']} mm² of the marked opening lies outside the patch.")
    if 0 <= placement["margin_mm"] < TIGHT_MARGIN_MM:
        notes.append(f"The patch overhangs the opening by only {placement['margin_mm']} mm on its tight side.")
    if placement["outside_photo_percent"] > 0:
        notes.append(
            f"{placement['outside_photo_percent']}% of the patch falls outside the band photograph; "
            "overhang beyond the photographed band is not judged."
        )
    if cut["cloth_margin_mm"] < 1.0:
        notes.append(f"The cut sits only {cut['cloth_margin_mm']} mm inside the lining's visible cloth.")
    if chosen["grain_mismatch_short_degrees"] > 5.0:
        notes.append(
            f"The second yarn direction lands {chosen['grain_mismatch_short_degrees']}° off the old band."
        )
    if opening["unsupported_percent"] > 0:
        notes.append(f"{opening['unsupported_percent']}% of the marked opening is outside the photographed band.")
    if blue and blue.get("edges"):
        joined = [item for item in blue["edges"] if item.get("angle_difference") is not None]
        for edge in joined:
            if edge["angle_difference"] > 3.0 or edge["offset_step_mm"] > 0.6:
                notes.append(
                    f"Blue {edge['edge']}: the run before the break and the run after it differ by "
                    f"{edge['offset_step_mm']} mm across the band and {edge['angle_difference']}° in line angle. "
                    "Bring them together over this piece."
                )
            else:
                notes.append(
                    f"Blue {edge['edge']}: the runs either side of the break agree to {edge['offset_step_mm']} mm "
                    f"and {edge['angle_difference']}°; keep that line straight when joining."
                )
        if not joined:
            notes.append(
                "Blue edging is seen on one side of the break only, so its two ends cannot be compared here; "
                "the plain lining supplies no blue of its own."
            )
        else:
            notes.append("The plain lining has no blue edging of its own; the old blue has to meet over this piece.")
        pieces = [item for edge in blue["edges"] for item in (edge.get("before"), edge.get("after")) if item]
        if pieces and any(item["angle_from_long"] > 5.0 for item in pieces):
            notes.append(
                "Part of the blue edging does not run parallel to the long yarn direction; "
                "follow the old blue when the seam is stitched, not the long grain alone."
            )
    notes.append(
        "The cut is a plain rectangle square to the lining's own yarn direction; "
        "the old blue edging is not rebuilt by this piece."
    )
    return notes


def plan_overlay(band, opening, chosen, blue):
    image = base_rgba(band).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = image.size
    span = extent_mm(band)
    stroke = max(2, round(min(width, height) / 260))
    opening_points = [(float(x), float(y)) for x, y in opening["mm"] / span * [width, height]]
    draw.polygon(opening_points, fill=OPENING_COLOR + (76,))
    corners = np.asarray(chosen["placement"]["corners_mm"]) / span * [width, height]
    patch = [(float(x), float(y)) for x, y in corners]
    draw.polygon(patch, fill=PATCH_COLOR + (26,))
    image = Image.alpha_composite(image, layer)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.line(opening_points + [opening_points[0]], fill=OPENING_COLOR, width=stroke, joint="curve")
    draw.line(patch + [patch[0]], fill=PATCH_COLOR, width=stroke, joint="curve")
    center = np.asarray(chosen["placement"]["center_mm"]) / span * [width, height]
    draw.ellipse((center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=INK, outline=SHADOW)
    arrow = rotate(np.array([0.0, -1.0]), chosen["_rotation"]) * min(width, height) * 0.08
    draw.line([tuple(center), tuple(center + arrow)], fill=CLOTH_COLOR, width=max(2, stroke))
    draw.text(
        tuple(center + arrow + [4, -6]), "piece up", fill=INK, stroke_width=2, stroke_fill=SHADOW,
        font=font(max(11, round(min(width, height) / 42))),
    )
    if blue and blue.get("edges"):
        for edge in blue["edges"]:
            for side in ("before", "after"):
                part = edge.get(side)
                if part:
                    point = np.asarray(part["centroid_mm"]) / span * [width, height]
                    draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4),
                                 outline=BLUE_COLOR, width=2)
    return image


def cloth_overlay(lining, chosen):
    image = base_rgba(lining).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = image.size
    span = extent_mm(lining)
    stroke = max(2, round(min(width, height) / 240))
    corners = np.asarray(chosen["cut"]["corners_mm"]) / span * [width, height]
    patch = [(float(x), float(y)) for x, y in corners]
    draw.polygon(patch, fill=PATCH_COLOR + (30,))
    image = Image.alpha_composite(image, layer)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.line(patch + [patch[0]], fill=PATCH_COLOR, width=stroke, joint="curve")
    center = np.asarray(chosen["cut"]["center_mm"]) / span * [width, height]
    up = rotate(np.array([0.0, -1.0]), chosen["_rotation"]) * min(width, height) * 0.18
    draw.line([tuple(center), tuple(center + up)], fill=CLOTH_COLOR, width=stroke)
    draw.ellipse((center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=INK, outline=SHADOW)
    draw.text(
        (center[0] + 6, center[1] + 6),
        f"cut 28 × 18 mm · marked up end → {chosen['up']['phrase']}",
        fill=INK, stroke_width=2, stroke_fill=SHADOW,
        font=font(max(11, round(min(width, height) / 40))),
    )
    return image


def seam_classifier(band, lining, opening, chosen):
    band_states = cloth_states(
        band,
        float(chosen.get("band_grain_pitch") or 1.0),
        region=chosen.get("band_region"),
        exclude=opening["mask"],
    )
    lining_states = cloth_states(lining, float(chosen.get("lining_pitch_mm") or 1.0))
    band_span = extent_mm(band)
    cloth_span = extent_mm(lining)
    center = np.asarray(chosen["placement"]["center_mm"])
    cut_center = np.asarray(chosen["cut"]["center_mm"])
    along = unit(chosen["long_band_angle"])
    across = unit(chosen["long_band_angle"] + 90.0)
    rotation = chosen["_rotation"]
    width, height = band.extent

    def read(point_mm):
        local = np.array([(point_mm - center) @ along, (point_mm - center) @ across])
        col = point_mm[0] / band.scale[0]
        row = point_mm[1] / band.scale[1]
        in_photo = 0 <= col < width and 0 <= row < height
        if not in_photo:
            return UNSEEN_BAND
        if opening["mask"][int(row), int(col)]:
            return GAP
        if abs(local[0]) > PATCH_LENGTH_MM / 2 or abs(local[1]) > PATCH_WIDTH_MM / 2:
            return OFF_PATCH
        if not photo_support(band)[int(row), int(col)]:
            return UNSEEN_BAND
        if band_states["unclear"][int(row), int(col)]:
            return UNCLEAR_BAND
        if band_states["smooth"][int(row), int(col)]:
            return NOT_BAND_CLOTH
        cloth_point = rotate(point_mm - center, -rotation) + cut_center
        if np.any(cloth_point < 0) or np.any(cloth_point >= cloth_span):
            return OUTSIDE_CLOTH
        ccol = int(cloth_point[0] / lining.scale[0])
        crow = int(cloth_point[1] / lining.scale[1])
        if lining_states["unclear"][crow, ccol]:
            return UNSEEN_CLOTH
        if lining_states["smooth"][crow, ccol]:
            return NOT_LINING_CLOTH
        return COUNTED

    return read


def measure_seam(band, lining, opening, chosen, points, step=SEAM_STEP_MM):
    classify = seam_classifier(band, lining, opening, chosen)
    positions = np.asarray(points, dtype=np.float64) * extent_mm(band)
    segments = []
    totals = {status: 0.0 for status in STATUS_ORDER}
    counted_positions = []
    travelled = 0.0
    for index, (start, stop) in enumerate(zip(positions, positions[1:]), 1):
        length = float(np.linalg.norm(stop - start))
        if length < 1e-9:
            raise InputError("Each seam segment needs two distinct points.")
        count = max(1, int(math.ceil(length / step)))
        mids = start + (np.arange(count) + 0.5)[:, None] * (stop - start) / count
        states = [classify(point) for point in mids]
        stretch = length / count
        row = {status: 0.0 for status in STATUS_ORDER}
        for offset, status in enumerate(states):
            row[status] += stretch
            if status == COUNTED:
                counted_positions.append(travelled + (offset + 0.5) * stretch)
        travelled += length
        for status in STATUS_ORDER:
            totals[status] += row[status]
        segments.append({
            "index": index,
            "length_mm": round(length, 4),
            "counted_mm": round(row[COUNTED], 4),
            "stretches_mm": {status: round(value, 4) for status, value in row.items()},
            "fractions": {status: round(value / length, 4) for status, value in row.items()},
            "start_mm": float_list(start, 3),
            "end_mm": float_list(stop, 3),
            "dominant": max(STATUS_ORDER, key=lambda status: row[status]),
        })
    total_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    runs = []
    for value in counted_positions:
        if runs and value - runs[-1][1] <= step * 1.01:
            runs[-1][1] = value
        else:
            runs.append([value, value])
    return {
        "unit": "mm",
        "segments": segments,
        "total_length_mm": round(total_length, 4),
        "counted_mm": round(totals[COUNTED], 4),
        "summary_mm": {status: round(value, 4) for status, value in totals.items()},
        "counted_runs_mm": [[round(first, 4), round(last, 4)] for first, last in runs],
        "lead_in_mm": round(runs[0][0], 4) if runs else None,
        "trail_out_mm": round(total_length - runs[-1][1], 4) if runs else None,
        "station_step_mm": step,
        "note": (
            "A stretch counts only where the same millimetre is visible as old band, lies on the placed "
            "28 × 18 mm cut, and is visible as cloth in the lining photograph. Millimetres over the opening, "
            "off the patch, or outside either photograph are listed separately and never added to the seam."
        ),
    }


def seam_overlay(band, opening, chosen, points, metrics, classify):
    image = base_rgba(band).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    span = extent_mm(band)
    stroke = max(2, round(min(width, height) / 280))
    opening_points = [(float(x), float(y)) for x, y in opening["mm"] / span * [width, height]]
    draw.line(opening_points + [opening_points[0]], fill=OPENING_COLOR, width=stroke, joint="curve")
    corners = np.asarray(chosen["placement"]["corners_mm"]) / span * [width, height]
    patch = [(float(x), float(y)) for x, y in corners]
    draw.line(patch + [patch[0]], fill=PATCH_COLOR, width=stroke, joint="curve")
    positions = np.asarray(points, dtype=np.float64) * extent_mm(band)
    for start, stop in zip(positions, positions[1:]):
        length = float(np.linalg.norm(stop - start))
        count = max(1, int(math.ceil(length / 0.12)))
        mids = start + (np.arange(count) + 0.5)[:, None] * (stop - start) / count
        for point in mids:
            status = classify(point)
            pixels = point / span * [width, height]
            draw.ellipse(
                (pixels[0] - stroke * 0.7, pixels[1] - stroke * 0.7,
                 pixels[0] + stroke * 0.7, pixels[1] + stroke * 0.7),
                fill=STATUS_COLORS[status],
            )
    for row, (start, stop) in zip(metrics["segments"], zip(positions, positions[1:])):
        a = start / span * [width, height]
        b = stop / span * [width, height]
        draw.line([tuple(a), tuple(b)], fill=SHADOW, width=1)
        middle = (a + b) / 2
        draw.text(
            tuple(middle + [4, -12]),
            f"{row['counted_mm']:.2f} mm counted",
            fill=INK, stroke_width=2, stroke_fill=SHADOW,
            font=font(max(10, round(min(width, height) / 50))),
        )
    return image


def region_from_request(tile, value, label):
    if value is None:
        return None
    chosen = identity(value, label)
    found = next((item for item in tile["regions"] if item["id"] == chosen), None)
    if found is None:
        raise InputError(f"The {label.lower()} is no longer on that specimen.")
    if found["purpose"] == "ignore":
        raise InputError(f"The {label.lower()} is marked as an exclusion.")
    return found


def lining_entry(store, tile_id, up_angle, cloth_region_id):
    tile = store.tile(identity(tile_id, "Lining specimen ID"))
    frame = specimen_frame(store.blobs, tile)
    if not frame.scale or not tile.get("width_mm") or not tile.get("height_mm"):
        raise InputError(f"Give {tile['name']} a photographed width and height in millimetres first.")
    if not tile.get("plane"):
        raise InputError(f"Mark the photographed plane on {tile['name']} first.")
    region = region_from_request(tile, cloth_region_id, "Lining cloth region")
    if up_angle is None:
        raise InputError(f"Mark which end of {tile['name']} is up.")
    base_angle = float(up_angle)
    rough, detected = cloth_mask(frame, 1.0, region)
    grain, crossing = grain_axes(frame, rough)
    cloth, detected = cloth_mask(frame, float(np.mean([item['pitch_mm'] for item in grain])), region)
    refined, crossing = grain_axes(frame, cloth)
    # The marked "up" line should follow one of the two measured yarn families;
    # the family it follows is the one that runs up the piece.
    distances = [angle_difference(base_angle, item["angle"]) for item in refined]
    index = int(np.argmin(distances))
    snapped = refined[index]["angle"] if distances[index] <= 30.0 else float(base_angle % 180.0)
    snapped = float(snapped % 180.0)
    entry = {
        "tile_id": tile["id"],
        "name": tile["name"],
        "photo_id": frame.photo["id"],
        "photo_name": frame.photo["name"],
        "plane_mm": [round(float(value), 3) for value in extent_mm(frame)],
        "grain": refined,
        "crossing_angle": crossing,
        "up_marked_angle": round(float(up_angle), 2),
        "up_angle": round(snapped, 2),
        "up_angle_snapped_to_grain": bool(distances[index] <= 30.0),
        "up_grain_index": index,
        "up_phrase": compass(base_angle),
        "cloth_mask_detected": bool(detected),
        "cloth_region_id": region["id"] if region else None,
        "cloth_support_percent": round(100.0 * np.count_nonzero(cloth) / max(int(np.count_nonzero(photo_support(frame))), 1), 2),
        "cloth_pitch_mm": round(float(np.mean([item["pitch_mm"] for item in refined])), 3),
    }
    return tile, frame, cloth, entry


def create_plan(store, specimen_id, payload):
    data = dictionary(payload, "Patch plan")
    title = string(data.get("title"), "Plan title", maximum=90)
    band = store.tile(identity(specimen_id, "Specimen ID"))
    frame = specimen_frame(store.blobs, band)
    if not frame.scale or not band.get("width_mm") or not band.get("height_mm"):
        raise InputError("Give this specimen a photographed width and height in millimetres first.")
    opening_region = region_from_request(band, data.get("opening_region_id"), "Opening region")
    if opening_region is None:
        raise InputError("Choose the region that outlines the opening.")
    band_region = region_from_request(band, data.get("band_region_id"), "Band region")
    raw_linings = collection(data.get("linings"), "Lining pieces", minimum=1, maximum=2)
    parsed = []
    for item in raw_linings:
        record = dictionary(item, "Lining piece")
        angle = record.get("up_angle_degrees")
        if angle is None:
            raise InputError("Mark which end of each lining piece is up.")
        tile_id = identity(record.get("tile_id"), "Lining specimen ID")
        if tile_id == band["id"]:
            raise InputError("A lining piece has to be a different specimen from the broken band.")
        parsed.append((tile_id, scalar(angle, "Marked up end", lower=0, upper=179), record.get("cloth_region_id")))
    if len(parsed) == 2 and parsed[0][0] == parsed[1][0]:
        raise InputError("Choose two different lining specimens.")
    band_cloth, band_detected = cloth_mask(frame, 1.0, band_region)
    band_grain, band_crossing = grain_axes(frame, band_cloth)
    band_cloth, band_detected = cloth_mask(frame, float(np.mean([item["pitch_mm"] for item in band_grain])), band_region)
    band_grain, band_crossing = grain_axes(frame, band_cloth)
    opening = opening_metrics(frame, opening_region)
    blue = blue_edge(frame, opening, band_grain[0]["angle"], band_cloth)
    long_axis = long_direction(frame, band_cloth, band_grain, blue)
    blue = blue_edge(frame, opening, long_axis["angle"], band_cloth) if blue else blue
    candidates = []
    lining_frames = []
    lining_records = []
    for tile_id, up_angle, region_id in parsed:
        try:
            tile, lining_frame, cloth, entry = lining_entry(store, tile_id, up_angle, region_id)
        except InputError as error:
            tile = store.tile(identity(tile_id, "Lining specimen ID"))
            lining_records.append({
                "tile_id": tile["id"],
                "name": tile["name"],
                "photo_id": None,
                "grain": [],
                "crossing_angle": None,
                "up_marked_angle": round(float(up_angle), 2),
                "up_angle": round(float(up_angle % 180.0), 2),
                "cloth_pitch_mm": 1.0,
                "error": str(error),
            })
            continue
        up_angle = entry["up_angle"]
        lining_frames.append((tile, lining_frame, cloth, entry))
        lining_records.append(entry)
        for cloth_index in (0, 1):
                rotation = float(wrap180(long_axis["angle"] - entry["grain"][cloth_index]["angle"]))
                try:
                    candidate = build_candidate(
                        frame, lining_frame, opening, long_axis, entry["grain"], rotation, cloth_index,
                        up_angle, tile["id"], tile["name"],
                    )
                except InputError as error:
                    candidate = {
                        "lining_tile_id": tile["id"],
                        "lining_name": tile["name"],
                        "lining_photo_id": lining_frame.photo["id"],
                        "rotation_clockwise_degrees": round(rotation, 2),
                        "long_band_angle": round(long_axis["angle"], 2),
                        "other_band_angle": round(long_axis["other_angle"], 2),
                        "long_lining_angle": entry["grain"][cloth_index]["angle"],
                        "grain_mismatch_long_degrees": round(angle_difference(entry["grain"][cloth_index]["angle"] + rotation, long_axis["angle"]), 2),
                        "grain_mismatch_short_degrees": round(angle_difference(entry["grain"][1 - cloth_index]["angle"] + rotation, long_axis["other_angle"]), 2),
                        "grain_mismatch_total_degrees": None,
                        "up": {"angle": None, "phrase": "unplaced"},
                        "feasible": False,
                        "reason": str(error),
                        "cloth_mask_detected": False,
                        "placement": None,
                        "cut": None,
                        "_rotation": rotation,
                    }
                    candidate["grain_mismatch_total_degrees"] = round(
                        candidate["grain_mismatch_long_degrees"] + candidate["grain_mismatch_short_degrees"], 2)
                candidate["band_grain_pitch"] = float(np.mean([item["pitch_mm"] for item in band_grain]))
                candidate["lining_pitch_mm"] = entry["cloth_pitch_mm"]
                candidates.append(candidate)
    if not candidates:
        details = "; ".join(item["error"] for item in lining_records if item.get("error"))
        raise InputError(f"No lining piece could be read. {details or 'Show both yarn directions on each piece.'}")
    chosen = choose(candidates)
    if not chosen.get("placement") or not chosen.get("cut"):
        details = "; ".join(item.get("reason") for item in candidates if item.get("reason"))
        raise InputError(
            "No reading of this lining could be placed. " + (details or "Show both yarn directions on the lining.")
        )
    chosen["long_band_angle"] = round(long_axis["angle"], 2)
    chosen["band_grain"] = band_grain
    chosen["blue"] = blue
    chosen["chosen"] = True
    # Keep the band's cloth region beside the chosen piece so a later seam trace
    # measures the same cloth the plan used.
    chosen["band_region"] = band_region
    chosen["band_grain_pitch"] = float(np.mean([item["pitch_mm"] for item in band_grain]))
    plan = {
        "title": title,
        "tile_id": band["id"],
        "specimen": specimen_provenance(band),
        "source": capture_provenance(frame),
        "opening": {key: value for key, value in opening.items() if key not in ("mask", "mm")},
        "band": {
            "grain": band_grain,
            "crossing_angle": band_crossing,
            "cloth_mask_detected": bool(band_detected),
            "cloth_region_id": band_region["id"] if band_region else None,
            "plane_mm": [round(float(value), 3) for value in extent_mm(frame)],
            "long_angle": long_axis["angle"],
            "long_note": long_axis["note"],
            "long_extent_mm": long_axis["long_extent_mm"],
            "short_extent_mm": long_axis["short_extent_mm"],
        },
        "blue": blue,
        "linings": lining_records,
        "chosen": public(chosen),
        "alternatives": [public(item) for item in candidates if item is not chosen],
        "caveats": caveats(chosen, opening, blue),
        "patch_mm": [PATCH_LENGTH_MM, PATCH_WIDTH_MM],
        "images": {},
    }
    chosen_lining = next(item for item in lining_frames if item[1].photo["id"] == chosen["lining_photo_id"])
    plan["images"] = {
        "plan": stored_preview(plan_overlay(frame, opening, chosen, blue), store.blobs),
        "cut": stored_preview(cloth_overlay(chosen_lining[1], chosen), store.blobs),
    }
    plan["seam_ids"] = []
    record = store.save_mend(plan)
    return store.mend(record["id"])


def seam_shape(store, mend_id, points, *, plan=None, band=None):
    """Measure a repair seam; shared by the plan runner and the Geometry trace."""
    plan = plan if plan is not None else store.mend(identity(mend_id, "Mend plan ID"))
    band = band if band is not None else store.tile(plan["tile_id"])
    frame = specimen_frame(store.blobs, band)
    lining_tile = store.tile(plan["chosen"]["lining_tile_id"])
    lining = specimen_frame(store.blobs, lining_tile)
    opening_region = next(
        (item for item in band["regions"] if item["id"] == plan["opening"]["region_id"]), None
    )
    if opening_region is None:
        raise InputError("The opening region used by this plan has been removed.")
    opening = opening_metrics(frame, opening_region)
    chosen = dict(plan["chosen"])
    chosen["_rotation"] = chosen["rotation_clockwise_degrees"]
    chosen["band_grain_pitch"] = float(np.mean([item["pitch_mm"] for item in plan["band"]["grain"]]))
    if not chosen.get("band_region") and plan["band"].get("cloth_region_id"):
        chosen["band_region"] = next(
            (item for item in band["regions"] if item["id"] == plan["band"]["cloth_region_id"]), None
        )
    lining_record = next(item for item in plan["linings"] if item["tile_id"] == lining_tile["id"])
    chosen["lining_pitch_mm"] = lining_record["cloth_pitch_mm"]
    metrics = measure_seam(frame, lining, opening, chosen, points)
    classify = seam_classifier(frame, lining, opening, chosen)
    overlay = seam_overlay(frame, opening, chosen, points, metrics, classify)
    cumulative = [0.0]
    for item in metrics["segments"]:
        cumulative.append(round(cumulative[-1] + item["length_mm"], 4))
    return {
        "tile_id": band["id"],
        "mend_id": plan["id"],
        "specimen": specimen_provenance(band),
        "source": capture_provenance(frame),
        "region": opening_region,
        "plan_snapshot": {
            "id": plan["id"],
            "title": plan["title"],
            "patch_mm": plan["patch_mm"],
            "chosen": plan["chosen"],
            "opening": plan["opening"],
        },
        "shape": {
            "points": points,
            "seam": metrics,
            "cumulative_mm": cumulative,
            "length_unit": "mm",
        },
        "image": stored_preview(overlay, store.blobs),
    }


def create_seam(store, mend_id, payload):
    data = dictionary(payload, "Seam trace")
    title = string(data.get("title"), "Seam title", maximum=90)
    points = [
        numeric_pair(item, "Seam point", lower=0, upper=1)
        for item in collection(data.get("points"), "Seam points", minimum=2, maximum=120)
    ]
    record = {"title": title, "mode": "seam", **seam_shape(store, mend_id, points)}
    saved = store.save_metrology(record)
    seams = store.mend(record["mend_id"])
    seams["seam_ids"] = [*seams.get("seam_ids", []), saved["id"]]
    store.update_mend(seams)
    return saved
