"""Broken ribbon fragment seam workbench.

Two photographs of the two pieces of one broken woven ribbon are rectified
into a shared physical band plane.  The right fragment may only be shifted one
notch up, one notch down, or left on the middle setting.  A notch is chosen
from evidence actually visible in the captures: the continuity of the blue
edge stripes and the pairings of frayed warp ends.

Nothing outside the photographed fragment outlines is invented.  Gaps stay
empty and line-scan stations that do not land on a real fragment pixel are
returned blank with provenance back to the originating fragment photo.
"""

from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .color import float_list, linear_lab
from .geometry import homography, line_grid, polygon_mask
from .model import InputError, collection, dictionary, numeric_pair, scalar
from .photos import linear_to_srgb, luminance, srgb_to_linear, uint8_image


SIDES = ("left", "right")
NOTCHES = (-1, 0, 1)
NOTCH_LABEL = {-1: "one grid up", 0: "middle grid", 1: "one grid down"}
MAX_OUTLINE = 60
MIN_OUTLINE = 6
MIN_BAND_EDGE = 40
BLUE_B_MARGIN = 6.0
THREAD_PIXEL_MIN = 3
THREAD_PIXEL_MAX = 260
MATCH_TOLERANCE_FACTOR = 0.65
GAP_PADDING_PX = 6
BACKGROUND_LINEAR = np.array([0.075, 0.078, 0.074], dtype=np.float32)
STITCH_ACCENT = (246, 139, 96)
STITCH_GOOD = (126, 201, 147)
STITCH_WARN = (233, 183, 92)
STITCH_INK = (255, 245, 225)
STITCH_SHADOW = (24, 30, 33)


@dataclass(frozen=True)
class Fragment:
    side: str
    rgb: np.ndarray            # sRGB 0..1, rectified band plane, H x W x 3
    linear: np.ndarray         # linear RGB, H x W x 3
    support: np.ndarray        # bool, inside the photographed outline
    tear: np.ndarray           # int, tear x per band row (length H); NaN-safe via -1
    source_to_band: tuple      # 3x3 homography source-photo -> band frame
    band_to_source: tuple      # inverse homography
    photo: dict
    pixels_per_mm: float

    @property
    def height(self):
        return self.rgb.shape[0]

    @property
    def width(self):
        return self.rgb.shape[1]

    def tear_x(self):
        rows = np.flatnonzero(self.tear >= 0)
        return int(round(float(np.median(self.tear[rows])))) if len(rows) else 0


def _homography_matrix(unit_target, source):
    return homography(unit_target, source)


def _warp_photo_to_band(photo_rgb, photo_support, matrix, width, height):
    source_height, source_width = photo_support.shape
    target_x = (np.arange(width, dtype=np.float32)[None, :] + 0.5) / width
    rgb_linear = srgb_to_linear(photo_rgb)
    weighted = rgb_linear * photo_support[:, :, None]
    linear = np.zeros((height, width, 3), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=bool)
    for start in range(0, height, 128):
        stop = min(start + 128, height)
        target_y = (np.arange(start, stop, dtype=np.float32)[:, None] + 0.5) / height
        denominator = matrix[2, 0] * target_x + matrix[2, 1] * target_y + matrix[2, 2]
        src_u = (matrix[0, 0] * target_x + matrix[0, 1] * target_y + matrix[0, 2]) / denominator
        src_v = (matrix[1, 0] * target_x + matrix[1, 1] * target_y + matrix[1, 2]) / denominator
        # Convert normalized source coordinates to pixel-center sampling coordinates.
        src_x = src_u * source_width - 0.5
        src_y = src_v * source_height - 0.5
        within = (src_x >= -0.5) & (src_x <= source_width - 0.5) & (src_y >= -0.5) & (src_y <= source_height - 0.5)
        coords = (src_y, src_x)
        opacity = ndimage.map_coordinates(photo_support.astype(np.float32), coords, order=1, mode="constant", cval=0)
        present = within & (opacity >= 0.995)
        coverage[start:stop] = present
        block = linear[start:stop]
        for channel in range(3):
            samples = ndimage.map_coordinates(weighted[:, :, channel], coords, order=1, mode="constant", cval=0)
            np.divide(samples, np.maximum(opacity, 1e-6), where=present, out=block[:, :, channel])
    display = linear_to_srgb(linear)
    return display, linear, coverage


def _outline_mask(width, height, vertices_band):
    return polygon_mask((width, height), vertices_band)


def rectify_fragment(side, photo, rgb, support, corners, outline, band_width, band_height, pixels_per_mm):
    """Warp one fragment photograph into the normalized band frame."""
    if band_width < MIN_BAND_EDGE or band_height < MIN_BAND_EDGE:
        raise InputError("The ribbon band would be smaller than 40 pixels on a side.")
    # The warper samples source pixels from target coordinates, so the matrix
    # maps a normalized band point to its normalized position in the source photo.
    matrix = _homography_matrix([[0, 0], [1, 0], [1, 1], [0, 1]], corners)
    inverse = np.linalg.inv(matrix)
    display, linear, warped_support = _warp_photo_to_band(rgb, support, matrix, band_width, band_height)
    band_outline = [_project_point(inverse, point) for point in outline]
    outline_mask = _outline_mask(band_width, band_height, band_outline)
    fragment_support = warped_support & outline_mask
    if fragment_support.sum() < 200:
        raise InputError(f"The {side} fragment outline covers too few usable pixels.")
    tear = _tear_profile(fragment_support, side)
    return Fragment(
        side=side,
        rgb=display,
        linear=linear,
        support=fragment_support,
        tear=tear,
        source_to_band=tuple(matrix.ravel()),
        band_to_source=tuple(inverse.ravel()),
        photo=photo,
        pixels_per_mm=float(pixels_per_mm),
    )


def _project_point(matrix, point):
    u, v = point
    denominator = matrix[2, 0] * u + matrix[2, 1] * v + matrix[2, 2]
    return [
        (matrix[0, 0] * u + matrix[0, 1] * v + matrix[0, 2]) / denominator,
        (matrix[1, 0] * u + matrix[1, 1] * v + matrix[1, 2]) / denominator,
    ]


def _tear_profile(support, side):
    """Return the torn-edge x coordinate for each band row, -1 where absent."""
    height, width = support.shape
    profile = np.full(height, -1, dtype=np.int32)
    rows = np.flatnonzero(support.any(axis=1))
    if not len(rows):
        return profile
    for row in rows:
        cols = np.flatnonzero(support[row])
        profile[row] = int(cols.max()) if side == "left" else int(cols.min())
    # Smooth single-row notches so the torn edge stays a stable boundary.
    known = profile >= 0
    filled = ndimage.median_filter(np.where(known, np.maximum(profile, 0), 0), size=5)
    return np.where(known, filled, -1).astype(np.int32)


def blue_stripes(fragment):
    """Locate blue stripe groups on a fragment, measured in band rows.

    Blue is identified from the Lab blue-yellow axis relative to the local
    cloth color rather than an image quantile, so an extra short decoy
    stripe on one fragment cannot move the detection threshold.
    """
    lab = linear_lab(fragment.linear)
    support = fragment.support
    cloth_b = float(np.median(lab[support, 2]))
    # Indigo sits on the negative (blue) side of the blue-yellow axis; the
    # local cloth is warm and positive, so axis separation is the key test.
    blue = support & (lab[:, :, 2] < cloth_b - BLUE_B_MARGIN * 3.2)
    labeled, count = ndimage.label(blue, structure=np.ones((3, 3)))
    groups = []
    for label_index in range(1, count + 1):
        component = labeled == label_index
        rows, cols = np.nonzero(component)
        if len(rows) < 24:
            continue
        span_cols = cols.max() - cols.min() + 1
        # Edge stripes run longitudinally. A group cut off by the tear only
        # occupies the surviving part of the band, so accept a shorter span
        # when the group reaches the torn boundary.
        at_tear = np.any(np.abs(cols - fragment.tear[rows]) <= 3)
        min_span = fragment.width * (0.12 if at_tear else 0.35)
        if span_cols < min_span:
            continue
        groups.append({
            "label": f"B{len(groups) + 1}",
            "rows": int(rows.min()),
            "row_end": int(rows.max()),
            "y": float(np.median(rows)),
            "pixels": int(len(rows)),
            "x_mean": float(np.mean(cols)),
        })
    groups.sort(key=lambda item: item["y"])
    for index, item in enumerate(groups):
        item["label"] = f"B{index + 1}"
    return groups, blue


def _warp_end_mask(fragment, lab):
    """Frayed warp ends are pale strands that run out of the textile into the mount.

    The fragment outline stops at the textile edge, so a loose end is visible
    just *beyond* that edge against the darker mount.  Candidates are bright,
    longitudinally elongated components inside a narrow outer band at the tear;
    ordinary weave texture inside the body never reaches this outer band.
    """
    luma = luminance(fragment.linear)
    height, width = luma.shape
    median_level = float(np.median(luma[fragment.support]))
    mount_level = float(np.median(luma[~fragment.support])) if (~fragment.support).any() else median_level * 0.4
    # A warp end is distinctly paler than the mount and close to cloth brightness.
    strand_level = max(mount_level * 1.8, 0.12)
    bright = luma > strand_level
    # Build the outer tear band: rows that contain textile, columns just beyond the edge.
    reach = int(max(10, width * 0.03))
    outer = np.zeros_like(bright)
    rows_with_edge = np.flatnonzero(fragment.tear >= 0)
    for row in rows_with_edge:
        edge = int(fragment.tear[row])
        if fragment.side == "left":
            start, stop = edge + 1, min(width, edge + reach + 1)
        else:
            start, stop = max(0, edge - reach), edge
        if stop > start:
            outer[row, start:stop] = True
    strands = bright & outer
    # The first and last few band rows meet the photo border; bright material
    # there is a border artifact rather than a loose warp end.
    edge_guard = max(7, int(round(height * 0.06)))
    labeled, count = ndimage.label(strands, structure=np.ones((3, 3)))
    candidates = []
    for label_index in range(1, count + 1):
        component = labeled == label_index
        pixels = int(component.sum())
        # A loose end is a multi-pixel yarn; isolated bright pixels are mount
        # grain and edge aliasing. Large blobs are the photo border, not yarn.
        if pixels < 14 or pixels > THREAD_PIXEL_MAX:
            continue
        rows, cols = np.nonzero(component)
        if rows.min() < edge_guard or rows.max() >= height - edge_guard:
            continue
        height_span = rows.max() - rows.min() + 1
        width_span = cols.max() - cols.min() + 1
        # Loose ends run along the band-length axis (x) and stay thin in y.
        if width_span < 6 or width_span < height_span * 1.4:
            continue
        # Require the component to stay within the outer fray band rather than
        # floating farther out in the mount where labels and borders live.
        gap_to_edge = np.abs(cols - fragment.tear[rows])
        if float(np.median(gap_to_edge)) > reach * 0.75:
            continue
        candidates.append({
            "y_min": int(rows.min()),
            "y_max": int(rows.max()),
            "y": float(np.median(rows)),
            "x_tip": float(cols.max() if fragment.side == "left" else cols.min()),
            "pixels": pixels,
        })
    # One frayed end can break into several touching components; merge pieces
    # whose y ranges overlap or sit within a two-row gap of each other.
    candidates.sort(key=lambda item: item["y"])
    merged = []
    for item in candidates:
        if merged and item["y_min"] <= merged[-1]["y_max"] + 2:
            host = merged[-1]
            host["y_min"] = min(host["y_min"], item["y_min"])
            host["y_max"] = max(host["y_max"], item["y_max"])
            host["pixels"] += item["pixels"]
            host["y"] = (host["y"] + item["y"]) / 2
            host["x_tip"] = max(host["x_tip"], item["x_tip"]) if fragment.side == "left" else min(host["x_tip"], item["x_tip"])
        else:
            merged.append(dict(item))
    ends = []
    for item in merged:
        item["rows"] = [item["y_min"], item["y_max"]]
        ends.append(item)
    ends.sort(key=lambda item: item["y"])
    for index, item in enumerate(ends):
        item["label"] = f"{fragment.side[0].upper()}{index + 1}"
    return ends


def warp_ends(fragment):
    lab = linear_lab(fragment.linear)
    return _warp_end_mask(fragment, lab)


def _stripe_score(left_stripes, right_stripes, shift, height):
    """Penalty for blue stripe groups that fail to continue across the seam."""
    if not left_stripes or not right_stripes:
        return None, []
    pairs = []
    residual_total = 0.0
    used = set()
    for stripe in left_stripes:
        target_y = stripe["y"] + shift
        candidates = [
            (abs(other["y"] - target_y), other)
            for other in right_stripes
            if other["label"] not in used
        ]
        if not candidates:
            continue
        gap, other = min(candidates, key=lambda item: item[0])
        tolerance = max(4.0, (stripe["row_end"] - stripe["rows"]) * MATCH_TOLERANCE_FACTOR)
        matched = gap <= tolerance
        if matched:
            used.add(other["label"])
            residual_total += gap
        pairs.append({
            "left": stripe["label"],
            "right": other["label"] if matched else None,
            "residual_px": round(float(gap), 3),
            "matched": bool(matched),
        })
    matched_count = sum(1 for pair in pairs if pair["matched"])
    penalty = residual_total / max(1, matched_count) if matched_count else float(height)
    # Each unpaired stripe group is a hard discontinuity across the seam.
    discontinuities = len(left_stripes) - matched_count
    penalty += discontinuities * height * 0.25
    return penalty, pairs


def _thread_matches(left_threads, right_threads, shift):
    tolerance = _thread_tolerance(left_threads, right_threads)
    matches, unmatched_left, unmatched_right = [], set(), {item["label"] for item in right_threads}
    for thread in left_threads:
        target_y = thread["y"] + shift
        candidates = [
            (abs(other["y"] - target_y), other)
            for other in right_threads
            if other["label"] in unmatched_right
        ]
        if not candidates:
            unmatched_left.add(thread["label"])
            continue
        gap, other = min(candidates, key=lambda item: item[0])
        if gap <= tolerance:
            unmatched_right.discard(other["label"])
            matches.append({
                "left": thread["label"],
                "right": other["label"],
                "residual_px": round(float(gap), 3),
                "left_y": round(thread["y"], 3),
                "right_y_band": round(other["y"] - shift, 3),
            })
        else:
            unmatched_left.add(thread["label"])
    return matches, sorted(unmatched_left), sorted(unmatched_right), tolerance


def _thread_tolerance(left_threads, right_threads):
    positions = sorted(item["y"] for item in [*left_threads, *right_threads])
    gaps = np.diff(positions)
    gaps = gaps[gaps > 0.5]
    pitch = float(np.median(gaps)) if len(gaps) else 12.0
    return max(2.5, pitch * MATCH_TOLERANCE_FACTOR)


def _thread_score(matches, left_threads, right_threads, tolerance):
    possible = max(1, min(len(left_threads), len(right_threads)))
    if not matches:
        return float(len(left_threads) + len(right_threads)) * tolerance * 2.0
    residual = float(np.mean([item["residual_px"] for item in matches]))
    missing = (len(left_threads) + len(right_threads) - 2 * len(matches)) * tolerance
    return residual + missing / possible


def evaluate_notches(left, right, notch_step_px):
    """Score the three legal right-fragment placements from visible evidence."""
    left_stripes, _ = blue_stripes(left)
    right_stripes, _ = blue_stripes(right)
    left_threads = warp_ends(left)
    right_threads = warp_ends(right)
    report = []
    for notch in NOTCHES:
        shift = notch * notch_step_px
        stripe_penalty, stripe_pairs = _stripe_score(left_stripes, right_stripes, shift, left.height)
        matches, unmatched_left, unmatched_right, tolerance = _thread_matches(left_threads, right_threads, shift)
        thread_penalty = _thread_score(matches, left_threads, right_threads, tolerance)
        components = [value for value in (stripe_penalty, thread_penalty) if value is not None]
        total = float(np.mean(components)) if components else 0.0
        report.append({
            "notch": notch,
            "label": NOTCH_LABEL[notch],
            "shift_px": round(float(shift), 4),
            "score": round(total, 5),
            "stripe_penalty": None if stripe_penalty is None else round(stripe_penalty, 5),
            "thread_penalty": round(thread_penalty, 5),
            "stripe_pairs": stripe_pairs,
            "thread_matches": matches,
            "unmatched_left_threads": unmatched_left,
            "unmatched_right_threads": unmatched_right,
            "matched_thread_count": len(matches),
        })
    chosen = min(report, key=lambda item: item["score"])
    return chosen, report, {
        "left_stripes": left_stripes,
        "right_stripes": right_stripes,
        "left_threads": left_threads,
        "right_threads": right_threads,
        "thread_tolerance_px": round(tolerance, 4),
    }


def _band_layout(left, right, chosen_shift):
    """Place the surviving textile portions on either side of an empty seam.

    Each fragment is cropped at its own torn edge so the empty gap represents
    material that neither photograph shows; the gap is never filled in.
    """
    left_tear = left.tear_x()
    right_tear = right.tear_x()
    gap = GAP_PADDING_PX * 2
    left_keep = left_tear + 1                      # left fragment keeps columns 0..left_tear
    right_keep = right.width - right_tear          # right fragment keeps columns right_tear..end
    width = left_keep + gap + right_keep
    right_origin_y = int(round(chosen_shift))
    top = min(0, right_origin_y)
    bottom = max(left.height, right.height + right_origin_y)
    height = bottom - top
    # Crop rectangles stored as (start_col, end_col) in each fragment's own pixels.
    return {
        "width": int(width),
        "height": int(height),
        "left_offset": (0, -top),
        "right_offset": (left_keep + gap, right_origin_y - top),
        "left_crop": (0, left_keep),
        "right_crop": (right_tear, right.width),
        "gap": gap,
        "seam_x": int(left_tear + GAP_PADDING_PX),
        "left_tear_x": int(left_tear),
        "right_tear_x": int(right_tear),
    }


def composite_band(left, right, chosen_shift):
    """Assemble the stitched band without blending pixels across fragments."""
    layout = _band_layout(left, right, chosen_shift)
    width, height = layout["width"], layout["height"]
    canvas = np.broadcast_to(BACKGROUND_LINEAR, (height, width, 3)).copy()
    ownership = np.full((height, width), "", dtype=object)
    supported = np.zeros((height, width), dtype=bool)

    def paste(fragment, offset_x, offset_y, name, crop):
        crop_start, crop_end = crop
        for y in range(fragment.height):
            target_y = y + offset_y
            if not 0 <= target_y < height:
                continue
            local_cols = np.flatnonzero(fragment.support[y, crop_start:crop_end]) + crop_start
            if not len(local_cols):
                continue
            target_x = local_cols - crop_start + offset_x
            valid = (target_x >= 0) & (target_x < width)
            local_cols, target_x = local_cols[valid], target_x[valid]
            canvas[target_y, target_x] = fragment.linear[y, local_cols]
            ownership[target_y, target_x] = name
            supported[target_y, target_x] = True

    paste(left, *layout["left_offset"], "left", layout["left_crop"])
    paste(right, *layout["right_offset"], "right", layout["right_crop"])
    return canvas, ownership, supported, layout


def annotation_overlay(canvas_linear, ownership, layout, left, right, chosen, evidence, chosen_shift):
    image = uint8_image(linear_to_srgb(canvas_linear), np.ones(canvas_linear.shape[:2], dtype=np.float32)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    seam_x = layout["seam_x"]
    height = layout["height"]
    # The seam itself is drawn in the empty gap, never over fragment pixels.
    for y in range(0, height, 10):
        draw.line((seam_x - 2, y, seam_x - 2, min(y + 5, height)), fill=STITCH_WARN, width=2)
        draw.line((seam_x + 2, y + 5, seam_x + 2, min(y + 10, height)), fill=STITCH_WARN, width=2)

    left_dx, left_dy = layout["left_offset"]
    right_dx, right_dy = layout["right_offset"]
    left_crop_start = layout["left_crop"][0]
    right_crop_start = layout["right_crop"][0]

    def thread_point(thread, offset_x, offset_y, crop_start):
        x = int(round(thread["x_tip"])) - crop_start + offset_x
        y = int(round(thread["y"])) + offset_y
        return x, y

    matched_pairs = {(item["left"], item["right"]) for item in chosen["thread_matches"]}
    left_lookup = {item["label"]: item for item in evidence["left_threads"]}
    right_lookup = {item["label"]: item for item in evidence["right_threads"]}
    # Joined warp ends: draw the pairing line across the empty gap only, so it
    # never paints over real fragment pixels.
    for left_label, right_label in sorted(matched_pairs):
        start = thread_point(left_lookup[left_label], left_dx, left_dy, left_crop_start)
        end = thread_point(right_lookup[right_label], right_dx, right_dy, right_crop_start)
        gap_start = min(layout["seam_x"] - GAP_PADDING_PX + 2, end[0])
        gap_end = max(layout["seam_x"] + GAP_PADDING_PX - 2, start[0])
        draw.line((gap_start, start[1], gap_end, end[1]), fill=STITCH_SHADOW, width=4)
        draw.line((gap_start, start[1], gap_end, end[1]), fill=STITCH_GOOD, width=2)
    # Unmatched frayed ends get a ring at their visible tip on their own fragment.
    for label in chosen["unmatched_left_threads"]:
        x, y = thread_point(left_lookup[label], left_dx, left_dy, left_crop_start)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=STITCH_ACCENT, width=2)
    for label in chosen["unmatched_right_threads"]:
        x, y = thread_point(right_lookup[label], right_dx, right_dy, right_crop_start)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=STITCH_ACCENT, width=2)

    # Mark blue stripe groups that do not continue, with a ring at each tear.
    for pair in chosen["stripe_pairs"]:
        if pair["matched"]:
            continue
        left_stripe = next(item for item in evidence["left_stripes"] if item["label"] == pair["left"])
        x_left = layout["left_tear_x"] + left_dx
        y_left = int(round(left_stripe["y"])) + left_dy
        draw.ellipse((x_left - 6, y_left - 6, x_left + 6, y_left + 6), outline=STITCH_ACCENT, width=2)
        if pair["right"]:
            right_stripe = next(item for item in evidence["right_stripes"] if item["label"] == pair["right"])
            x_right = right_dx
            y_right = int(round(right_stripe["y"])) + right_dy
            draw.ellipse((x_right - 6, y_right - 6, x_right + 6, y_right + 6), outline=STITCH_ACCENT, width=2)
    title = f"Chosen: {chosen['label']} · shift {chosen_shift:+.1f} px · green = joined warp · orange ring = no visible join"
    draw.text((8, 6), title, fill=STITCH_INK, stroke_width=2, stroke_fill=STITCH_SHADOW)
    return image


def _band_to_source_point(fragment, band_x, band_y):
    matrix = np.asarray(fragment.band_to_source, dtype=np.float64).reshape(3, 3)
    u = (band_x + 0.5) / fragment.width
    v = (band_y + 0.5) / fragment.height
    denominator = matrix[2, 0] * u + matrix[2, 1] * v + matrix[2, 2]
    src_u = (matrix[0, 0] * u + matrix[0, 1] * v + matrix[0, 2]) / denominator
    src_v = (matrix[1, 0] * u + matrix[1, 1] * v + matrix[1, 2]) / denominator
    photo = fragment.photo
    return [
        float(src_u * photo["width"]),
        float(src_v * photo["height"]),
    ]


def line_scan(left, right, layout, ownership, supported, start, end, stations, band_pixels):
    """Sample the user's line across the band; blank stations stay blank.

    Every color value is an average of real fragment pixels inside the
    transverse band.  Stations without a real pixel on either fragment are
    returned with null color and provenance, so the Line scan export keeps a
    visible gap instead of inventing a value.
    """
    extent = (layout["width"], layout["height"])
    coords, _ = line_grid(extent, start, end, stations, band_pixels)
    support_weight = ndimage.map_coordinates(
        supported.astype(np.float32), coords, order=1, mode="constant", cval=0
    )
    left_dx, left_dy = layout["left_offset"]
    right_dx, right_dy = layout["right_offset"]
    rows = []
    for index in range(stations):
        band_x = float(np.mean(coords[1][index]))
        band_y = float(np.mean(coords[0][index]))
        weights = support_weight[index]
        entry = {
            "station": index,
            "band_position": [round(band_x, 4), round(band_y, 4)],
            "fragment": None,
            "source_photo_id": None,
            "source_pixel": None,
            "lab": None,
            "intensity": None,
            "samples": int(round(float(weights.sum()))),
        }
        active = weights >= 0.999
        if not np.any(active):
            rows.append(entry)
            continue
        sample_y = coords[0][index][active]
        sample_x = coords[1][index][active]
        cols = np.rint(sample_x).astype(int)
        row_indices = np.rint(sample_y).astype(int)
        inside = (cols >= 0) & (cols < extent[0]) & (row_indices >= 0) & (row_indices < extent[1])
        cols, row_indices = cols[inside], row_indices[inside]
        owners = ownership[row_indices, cols]
        owners = owners[owners != ""]
        if not len(owners):
            rows.append(entry)
            continue
        side = "left" if np.count_nonzero(owners == "left") >= np.count_nonzero(owners == "right") else "right"
        fragment = left if side == "left" else right
        offset_x, offset_y = (left_dx, left_dy) if side == "left" else (right_dx, right_dy)
        crop_start = layout["left_crop"][0] if side == "left" else layout["right_crop"][0]
        local_x = cols - offset_x + crop_start
        local_y = row_indices - offset_y
        valid = (
            (local_x >= 0) & (local_x < fragment.width)
            & (local_y >= 0) & (local_y < fragment.height)
            & fragment.support[local_y, local_x]
        )
        local_x, local_y = local_x[valid], local_y[valid]
        if not len(local_x):
            rows.append(entry)
            continue
        colors = fragment.linear[local_y, local_x]
        source_point = _band_to_source_point(fragment, float(np.mean(local_x)), float(np.mean(local_y)))
        photo_width, photo_height = fragment.photo["width"], fragment.photo["height"]
        in_frame = (
            0 <= source_point[0] < photo_width
            and 0 <= source_point[1] < photo_height
        )
        if not in_frame:
            rows.append(entry)
            continue
        lab_value = linear_lab(colors).mean(axis=0)
        entry["fragment"] = side
        entry["source_photo_id"] = fragment.photo["id"]
        entry["source_pixel"] = [round(v, 4) for v in source_point]
        entry["lab"] = float_list(lab_value, 5)
        entry["intensity"] = round(float(luminance(colors).mean()), 6)
        rows.append(entry)
    if not any(item["fragment"] for item in rows):
        raise InputError("The drawn line never crosses a visible fragment pixel.")
    return {
        "start": start,
        "end": end,
        "stations": stations,
        "band_pixels": band_pixels,
        "rows": rows,
        "supported_stations": sum(1 for item in rows if item["fragment"]),
        "blank_stations": sum(1 for item in rows if not item["fragment"]),
        "note": "Blank stations fall in the unphotographed gap and carry no color value.",
    }


def parse_fragment_payload(side, data):
    record = dictionary(data, f"{side} fragment")
    photo_id = record.get("photo_id")
    if not isinstance(photo_id, str) or not photo_id:
        raise InputError(f"Choose the {side} fragment photograph.")
    corners = [numeric_pair(item, "Ribbon corner", lower=0, upper=1)
               for item in collection(record.get("corners"), "Ribbon corners", minimum=4, maximum=4)]
    outline = [numeric_pair(item, "Outline point", lower=0, upper=1)
               for item in collection(record.get("outline"), "Fragment outline", minimum=MIN_OUTLINE, maximum=MAX_OUTLINE)]
    return {"photo_id": photo_id, "corners": corners, "outline": outline}


def parse_scan_line(payload):
    data = dictionary(payload, "Line scan")
    start = numeric_pair(data.get("start"), "Line start", lower=0, upper=1.05)
    end = numeric_pair(data.get("end"), "Line end", lower=0, upper=1.05)
    stations = int(data.get("stations", 120))
    if not 12 <= stations <= 480:
        raise InputError("Stations must be between 12 and 480.")
    band_pixels = scalar(data.get("band_pixels", 3), "Band width px", lower=0, upper=24)
    return start, end, stations, band_pixels


def notch_step_from_payload(payload, default_px):
    if payload.get("notch_step_px") is None:
        return float(default_px)
    return scalar(payload.get("notch_step_px"), "Notch step px", lower=0.5, upper=400.0)
