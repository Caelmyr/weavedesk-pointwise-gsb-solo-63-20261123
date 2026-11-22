"""Raking-light pile relief estimation and sloped cover-edge length.

The workbench never treats a dark base color as a cast shadow.  A relief
feature is only accepted when its cast shadow can be located in *both*
raking-light captures on opposite sides of the feature, so the same physical
cluster is confirmed from two real photographs.  One workshop-wide scale
factor converts shadow length to pile height; it is calibrated against one
hand-measured pile and checked against the second.
"""

from dataclasses import dataclass
import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .art import INK, SHADOW, base_rgba, uint8_image
from .color import float_list
from .defects import gaussian_supported, robust_noise
from .geometry import polygon_mask
from .lighting import captured_pair, registration_pair, robust_exposure_ratio
from .model import InputError, collection, numeric_pair, scalar
from .photos import luminance


CONNECTIVITY = ndimage.generate_binary_structure(2, 2)
DEFAULT_MIN_SHADOW_PIXELS = 30
MIN_FEATURE_PIXELS = 8
MAX_COMPONENTS = 4000
RELIEF_VERSION = 1

# Palette used for the accepted height map (low to tall).
HEIGHT_STOPS = (
    (0.00, (34, 84, 116)),
    (0.35, (74, 150, 150)),
    (0.65, (196, 184, 96)),
    (1.00, (226, 128, 74)),
)
UNMEASURED_TINT = (0.20, 0.21, 0.22)
UNMEASURED_STRIPE = (0.55, 0.52, 0.47)
GAP_TINT = (224, 122, 110)
MEASURED_TINT = (126, 200, 168)


@dataclass(frozen=True)
class RakeView:
    role: str
    frame: object
    luminance: np.ndarray


def _scalar(value, label, lower, upper):
    return scalar(value, label, lower=lower, upper=upper)


def anchor_records(payload):
    """Validate the two hand-measured pile anchors supplied by the workshop."""
    raw = collection(payload.get("anchors") if isinstance(payload, dict) else payload,
                     "Pile calibration anchors", minimum=2, maximum=2)
    anchors = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise InputError(f"Pile anchor {index} must be an object.")
        vertices = collection(item.get("vertices"), f"Anchor {index} shadow outline", minimum=3, maximum=24)
        polygon = [numeric_pair(point, "Anchor outline point", lower=0, upper=1) for point in vertices]
        height = _scalar(item.get("pile_height_mm"), f"Anchor {index} measured pile height", 0.05, 200)
        label = str(item.get("label", f"Anchor {index}"))[:40]
        anchors.append({"label": label, "vertices": polygon, "pile_height_mm": float(height)})
    return anchors


def cover_points(payload):
    if isinstance(payload, dict):
        payload = payload.get("points")
    raw = collection(payload, "Cover edge points", minimum=2, maximum=45)
    return [numeric_pair(item, "Cover edge point", lower=0, upper=1) for item in raw]


def rake_views(store, tile):
    """Build the balanced left/right raking-light frame pair."""
    from .geometry import specimen_frame

    photos = {item["role"]: item for item in tile["photos"]}
    if "rake-left" not in photos or "rake-right" not in photos:
        raise InputError("Add and register one rake-left and one rake-right capture before estimating relief.")
    left = specimen_frame(store.blobs, tile, photos["rake-left"]["id"])
    right = specimen_frame(store.blobs, tile, photos["rake-right"]["id"])
    pairing = registration_pair(left, right)
    area = np.ones(left.extent[::-1], dtype=bool)
    common, fit_area, overlap = captured_pair(left, right, area)
    gain, balance = robust_exposure_ratio(luminance(left.linear), luminance(right.linear), fit_area)
    views = (
        RakeView("rake-left", left, luminance(left.linear)),
        RakeView("rake-right", right, luminance(right.linear) * gain),
    )
    return views, pairing, overlap, balance, gain


def diffuse_frame(store, tile):
    from .geometry import specimen_frame

    diffuse = next((item for item in tile["photos"] if item["role"] == "diffuse"), None)
    if diffuse is None:
        raise InputError("Add and register the diffuse front photograph; it is needed to reject dark base colors.")
    frame = specimen_frame(store.blobs, tile, diffuse["id"])
    if frame.scale is None:
        raise InputError("Enter the physical specimen dimensions so shadow lengths can be read in millimetres.")
    return frame


def local_darkness(frame_luminance, domain, sigma):
    """Local positive shadow depth relative to nearby lit ground.

    The baseline is a broad Gaussian average.  Shadow seed pixels are then
    expanded through contiguous pixels that are darker than that baseline, so
    the full interior of a wide uniform cast shadow is retained even though
    the Gaussian mean drifts down inside it.  A uniformly dark ground (deep
    base dye) never becomes a seed because its local depth stays at the noise
    floor.
    """
    broad, support = gaussian_supported(frame_luminance, domain.astype(np.float32), sigma)
    depth = np.maximum(broad - frame_luminance, 0.0)
    depth[~domain] = 0.0
    return depth, support


def merge_shadow_pieces(raw, height, width):
    """Merge fragments of one cast shadow split by a bright pile core.

    Fragments of the same physical shadow stay on the same cross-axis row and
    are separated along x only by the lit pile that interrupts the umbra.  Two
    fragments merge when their rows overlap and the x gap between them is no
    wider than the shadow band itself.
    """
    pending = sorted(raw, key=lambda item: (item["row_center"], item["x0"]))
    merged = []
    used = [False] * len(pending)
    for index, anchor in enumerate(pending):
        if used[index]:
            continue
        group = [anchor]
        used[index] = True
        changed = True
        while changed:
            changed = False
            for other_index, other in enumerate(pending):
                if used[other_index]:
                    continue
                if any(_same_shadow_band(member, other) for member in group):
                    group.append(other)
                    used[other_index] = True
                    changed = True
        x0 = min(item["x0"] for item in group)
        x1 = max(item["x1"] for item in group)
        rows = np.concatenate([
            np.nonzero(item["mask"])[0] + item["bounds"][0].start for item in group
        ])
        cols = np.concatenate([
            np.nonzero(item["mask"])[1] + item["bounds"][1].start for item in group
        ])
        local = np.zeros((height, width), dtype=bool)
        local[rows, cols] = True
        y_slice = slice(int(rows.min()), int(rows.max()) + 1)
        x_slice = slice(int(cols.min()), int(cols.max()) + 1)
        bounds = (y_slice, x_slice)
        mask = local[bounds]
        size = int(np.count_nonzero(mask))
        merged.append({
            "label_id": anchor["label_id"],
            "bounds": bounds,
            "mask": mask,
            "size": size,
            "fragment_count": len(group),
            "centroid": (float(cols.mean()), float(rows.mean())),
            "median_depth": float(np.median([item["median_depth"] for item in group])),
            "normalized_centroid": [float(cols.mean() / width), float(rows.mean() / height)],
        })
    return merged


def _same_shadow_band(first, second):
    # Row bands must actually overlap, not merely have nearby centers: a tall
    # fragment and a short fragment from different rows can share a center.
    first_top = first["row_center"] - first["row_span"] / 2
    first_bottom = first["row_center"] + first["row_span"] / 2
    second_top = second["row_center"] - second["row_span"] / 2
    second_bottom = second["row_center"] + second["row_span"] / 2
    overlap = min(first_bottom, second_bottom) - max(first_top, second_top)
    if overlap < 0.5 * min(first["row_span"], second["row_span"]):
        return False
    left, right = sorted((first, second), key=lambda item: item["x0"])
    gap = right["x0"] - left["x1"]
    # Overlapping x ranges mean the fragments abut inside one shadow band.
    if gap <= 0:
        return True
    band = min(first["row_span"], second["row_span"])
    return gap <= max(band * 3.0, 24.0)


def shadow_components(view, domain, sigma, sensitivity, min_pixels):
    depth, support = local_darkness(view.luminance, domain, sigma)
    eligible = domain & (support >= 0.75) & ~view.frame.clipped
    values = depth[eligible]
    if values.size < 80:
        raise InputError("Too little common raking-light area remains to locate cast shadows.")
    floor = robust_noise(values)
    threshold = max(0.012, (5.0 - 3.2 * sensitivity) * floor)
    seeds = eligible & (depth >= threshold)
    # Grow each shadow through contiguous pixels still clearly darker than the
    # broad local baseline.  This recovers the uniform interior of a wide cast
    # shadow whose middle the Gaussian baseline partially averaged down, while
    # deep base dye (excluded upstream) and faint mottling cannot enter.
    growable = eligible & (depth >= threshold * 0.45)
    seed_labels, seed_count = ndimage.label(seeds, CONNECTIVITY)
    if seed_count:
        grown = np.zeros_like(seeds)
        for label_id in range(1, seed_count + 1):
            region = seed_labels == label_id
            filled = ndimage.binary_propagation(region, mask=growable, structure=CONNECTIVITY)
            grown |= filled
        candidates = grown
    else:
        candidates = seeds
    labels, count = ndimage.label(candidates, CONNECTIVITY)
    if count > MAX_COMPONENTS:
        raise InputError("Too many shadow candidates. Lower the sensitivity or outline a smaller area.")
    objects = ndimage.find_objects(labels)
    height, width = view.luminance.shape
    raw = []
    for label_id, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        mask = labels[bounds] == label_id
        size = int(np.count_nonzero(mask))
        if size < min_pixels:
            continue
        rows, cols = np.nonzero(mask)
        rows += bounds[0].start
        cols += bounds[1].start
        raw.append({
            "label_id": label_id,
            "bounds": bounds,
            "mask": mask,
            "size": size,
            "row_center": float(rows.mean()),
            "row_span": float(rows.max() - rows.min() + 1),
            "x0": int(cols.min()),
            "x1": int(cols.max()) + 1,
            "centroid": (float(cols.mean()), float(rows.mean())),
            "median_depth": float(np.median(depth[rows, cols])),
            "normalized_centroid": [float(cols.mean() / width), float(rows.mean() / height)],
        })
    components = merge_shadow_pieces(raw, height, width)
    return components, depth, threshold, float(floor)


def ground_dark_mask(diffuse, domain, sigma):
    """Pixels that are already dark in diffuse front light.  Deep base dye is
    not a raking shadow and must be excluded."""
    lit = luminance(diffuse.linear)
    broad, support = gaussian_supported(lit, domain.astype(np.float32), sigma)
    eligible = domain & (support >= 0.7) & diffuse.valid
    values = lit[eligible]
    if values.size < 80:
        raise InputError("The diffuse photograph covers too little of the selected plane.")
    floor = robust_noise(values)
    # Local darkening that already exists under diffuse light, plus absolute
    # near-black pixels, is base coloration rather than a cast shadow.
    local_dark = (broad - lit) > max(2.6 * floor, 0.02)
    absolute_dark = lit < max(float(np.quantile(values, 0.04)), 0.02)
    return eligible & (local_dark | absolute_dark)


def anchor_shadow_reading(anchor, view, components, extent, domain):
    """Measure the shadow axis length inside a hand-drawn anchor outline."""
    outlined = polygon_mask(extent, anchor["vertices"]) & domain
    if np.count_nonzero(outlined) < MIN_FEATURE_PIXELS:
        raise InputError(f"Anchor {anchor['label']} outline does not cover enough registered pixels.")
    hits = []
    for component in components:
        rows, cols = np.nonzero(component["mask"])
        rows += component["bounds"][0].start
        cols += component["bounds"][1].start
        full = np.zeros(extent[::-1], dtype=bool)
        full[rows, cols] = True
        overlap = int(np.count_nonzero(full & outlined))
        if overlap:
            hits.append((overlap, component, full))
    if not hits:
        raise InputError(
            f"No raking-light cast shadow was detected inside anchor {anchor['label']}. "
            "Redraw the outline over the shadow that is visible in both raking captures."
        )
    hits.sort(key=lambda item: -item[0])
    overlap, component, full = hits[0]
    covered = int(np.count_nonzero(outlined))
    fraction = overlap / max(covered, 1)
    # The drawn outline should mostly coincide with a real detected shadow.
    # Length is taken from the full detected component so an outline that only
    # circles a recognizable middle stretch never truncates the shadow ruler.
    verified = fraction >= 0.35
    axis = shadow_axis_length(full, view.frame.scale)
    return {
        "component_label": component["label_id"],
        "detected_pixels": component["size"],
        "outline_pixels": covered,
        "overlap_pixels": overlap,
        "outline_overlap_percent": round(100 * fraction, 2),
        "verified_shadow": verified,
        "shadow_axis_mm": round(axis, 5),
        "centroid": component["normalized_centroid"],
    }


def shadow_axis_length(mask, scale):
    """Length of a shadow along the raking-light (x) axis in millimetres."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0.0
    extent_px = float(cols.max() - cols.min() + 1)
    if scale is None:
        return extent_px
    return extent_px * float(scale[0])


def choose_anchor(anchors):
    """Rank anchors by distance to the specimen edge."""
    positions = []
    for anchor in anchors:
        xs = [point[0] for point in anchor["vertices"]]
        ys = [point[1] for point in anchor["vertices"]]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        edge_distance = min(cx, 1 - cx, cy, 1 - cy)
        positions.append((edge_distance, cx, cy))
    return positions


def calibration_scales(anchors, anchor_metrics):
    """Per-anchor scale factor k = measured pile height / shadow length."""
    scales = []
    for anchor, metrics in zip(anchors, anchor_metrics):
        lengths = [metrics[0]["shadow_axis_mm"], metrics[1]["shadow_axis_mm"]]
        # The longest visible cast shadow gives the conservative (largest)
        # height estimate, so the tallest middle cluster is not under-called.
        shadow_mm = max(lengths)
        if shadow_mm <= 0:
            raise InputError(f"Anchor {anchor['label']} has no measurable shadow length.")
        factor = anchor["pile_height_mm"] / shadow_mm
        scales.append({
            "label": anchor["label"],
            "pile_height_mm": anchor["pile_height_mm"],
            "left_shadow_mm": metrics[0]["shadow_axis_mm"],
            "right_shadow_mm": metrics[1]["shadow_axis_mm"],
            "used_shadow_mm": round(shadow_mm, 5),
            "scale_factor": round(factor, 6),
            "left_verified": metrics[0]["verified_shadow"],
            "right_verified": metrics[1]["verified_shadow"],
        })
    return scales


def pair_features(left_components, right_components, match_tolerance_px, cross_axis_px):
    """Match the same physical cluster across the two opposite raking views.

    Opposite raking lights move cast shadows along the light axis (x).  A
    valid pair keeps almost the same cross-axis (y) position and stays within
    a normalized distance tolerance.  Components seen in only one view stay
    unmeasured: their identity cannot be confirmed from both real photos.
    """
    used_right = set()
    pairs = []
    for left in left_components:
        lx, ly = left["centroid"]
        best = None
        best_distance = None
        for right in right_components:
            if right["label_id"] in used_right:
                continue
            rx, ry = right["centroid"]
            dx, dy = rx - lx, ry - ly
            if abs(dy) > cross_axis_px:
                continue
            distance = math.hypot(dx, dy)
            if distance > match_tolerance_px or distance < 0.5:
                continue
            if best is None or distance < best_distance:
                best = right
                best_distance = distance
        if best is None:
            continue
        used_right.add(best["label_id"])
        pairs.append((left, best, float(best_distance)))
    return pairs


def embedded_component(component, shape):
    mask = np.zeros(shape, dtype=bool)
    rows, cols = np.nonzero(component["mask"])
    rows += component["bounds"][0].start
    cols += component["bounds"][1].start
    mask[rows, cols] = True
    return mask


def measure_relief(store, tile, payload):
    """End-to-end pile relief estimation from a raking pair plus a diffuse."""
    options = payload.get("options", {}) if isinstance(payload, dict) else {}
    sigma = _scalar(options.get("shading_radius", 7.0), "Shading radius px", 1.5, 40.0)
    sensitivity = _scalar(options.get("sensitivity", 0.62), "Shadow sensitivity", 0.05, 0.98)
    match_tolerance = _scalar(options.get("match_tolerance_px", 26.0), "Two-view match tolerance px", 2.0, 220.0)
    cross_axis = _scalar(options.get("cross_axis_px", 6.0), "Cross-axis alignment px", 0.5, 60.0)
    min_pixels = int(_scalar(options.get("min_shadow_pixels", DEFAULT_MIN_SHADOW_PIXELS),
                             "Minimum shadow footprint px", 8, 2000))
    forced = options.get("calibration_anchor")
    if forced is not None and forced not in (1, 2):
        raise InputError("Calibration anchor must be 1, 2, or left blank for automatic selection.")
    anchors = anchor_records(payload.get("anchors"))
    views, pairing, overlap, balance, gain = rake_views(store, tile)
    diffuse = diffuse_frame(store, tile)
    extent = views[0].frame.extent
    shape = extent[::-1]
    common = views[0].frame.valid & views[1].frame.valid & diffuse.valid
    domain = common & ~views[0].frame.clipped & ~views[1].frame.clipped
    if np.count_nonzero(domain) < 200:
        raise InputError("The three photographs share too little registered, unclipped specimen area.")

    base_dark = ground_dark_mask(diffuse, common, sigma)
    shadow_domain = domain & ~base_dark

    per_view = []
    thresholds = {}
    floors = {}
    for view in views:
        components, _depth, threshold, floor = shadow_components(
            view, shadow_domain, sigma, sensitivity, min_pixels,
        )
        per_view.append(components)
        thresholds[view.role] = threshold
        floors[view.role] = floor
    left_components, right_components = per_view

    # Per-anchor diagnostics in both views and the single workshop scale.
    anchor_metrics = []
    for anchor in anchors:
        rows = [
            anchor_shadow_reading(anchor, views[0], left_components, extent, shadow_domain),
            anchor_shadow_reading(anchor, views[1], right_components, extent, shadow_domain),
        ]
        anchor_metrics.append(rows)
    scales = calibration_scales(anchors, anchor_metrics)

    edge_positions = choose_anchor(anchors)
    if forced in (1, 2):
        chosen_index = forced - 1
        choice_reason = "forced by workshop request"
    else:
        # Nearest the image edge: protect the easily abraded border cluster.
        chosen_index = int(np.argmin([item[0] for item in edge_positions]))
        choice_reason = "anchor nearest the specimen edge"
    chosen = scales[chosen_index]
    other = scales[1 - chosen_index]
    factor = chosen["scale_factor"]

    # Cross-check with the second hand measurement under the single ruler.
    predicted_other = other["used_shadow_mm"] * factor
    cross_error_mm = predicted_other - other["pile_height_mm"]
    cross_error_percent = round(100 * cross_error_mm / other["pile_height_mm"], 3)
    disagreement = abs(cross_error_percent)

    # Surface every place the evidence is weak so the workshop can decide.
    warnings = []
    for scale in scales:
        if not (scale["left_verified"] and scale["right_verified"]):
            warnings.append(
                f"Anchor {scale['label']} outline only partly overlaps a detected shadow "
                "in one or both raking photos; redraw it over the recognizable shadow."
            )
    if disagreement > 30.0:
        warnings.append(
            f"The single ruler misses the second hand measurement by {cross_error_percent:.1f}%; "
            "the two anchor shadows are not consistent with one height scale."
        )
    elif disagreement > 15.0:
        warnings.append(
            f"Second anchor cross-check differs by {cross_error_percent:.1f}%; treat its height as approximate."
        )

    # Match physical clusters across the two raking views.
    pairs = pair_features(left_components, right_components, match_tolerance, cross_axis)
    height_field = np.zeros(shape, dtype=np.float32)
    measured_mask = np.zeros(shape, dtype=bool)
    feature_rows = []
    for index, (left, right, pair_distance) in enumerate(pairs, start=1):
        left_full = embedded_component(left, shape)
        right_full = embedded_component(right, shape)
        left_mm = shadow_axis_length(left_full, views[0].frame.scale)
        right_mm = shadow_axis_length(right_full, views[1].frame.scale)
        used_shadow = max(left_mm, right_mm)
        height = used_shadow * factor
        footprint = left_full | right_full
        height_field[footprint] = np.maximum(height_field[footprint], height)
        measured_mask |= footprint
        feature_rows.append({
            "feature": f"R{index:03d}",
            "centroid": float_list(np.asarray(left["normalized_centroid"]), 6),
            "left_shadow_mm": round(left_mm, 4),
            "right_shadow_mm": round(right_mm, 4),
            "used_shadow_mm": round(used_shadow, 4),
            "estimated_height_mm": round(height, 4),
            "pair_distance_px": round(pair_distance, 3),
            "pixels": int(np.count_nonzero(footprint)),
        })

    feature_rows.sort(key=lambda row: -row["estimated_height_mm"])
    for rank, row in enumerate(feature_rows, start=1):
        row["height_rank"] = rank

    unmeasured = domain & ~measured_mask
    # Mark dark ground that was deliberately excluded from shadow detection.
    excluded_dark = base_dark & domain

    metrics = {
        "method_version": RELIEF_VERSION,
        "pair": pairing,
        "support": overlap,
        "balance": balance,
        "gain": round(gain, 5),
        "shadow_threshold": round(thresholds["rake-left"], 5),
        "noise_floor": round(floors["rake-left"], 5),
        "scale_anchor": chosen["label"],
        "scale_choice_reason": choice_reason,
        "scale_factor_mm_per_mm_shadow": factor,
        "anchor_scales": scales,
        "cross_check": {
            "anchor": other["label"],
            "measured_height_mm": other["pile_height_mm"],
            "predicted_height_mm": round(predicted_other, 4),
            "error_mm": round(cross_error_mm, 4),
            "error_percent": cross_error_percent,
            "within_15_percent": disagreement <= 15.0,
            "within_30_percent": disagreement <= 30.0,
        },
        "matched_feature_count": len(feature_rows),
        "left_only_components": len(left_components) - len(pairs),
        "right_only_components": len(right_components) - len(pairs),
        "measured_pixels": int(np.count_nonzero(measured_mask)),
        "unmeasured_support_pixels": int(np.count_nonzero(unmeasured)),
        "dark_ground_excluded_pixels": int(np.count_nonzero(excluded_dark)),
        "features": feature_rows,
        "tallest_feature": feature_rows[0] if feature_rows else None,
        "warnings": warnings,
        "note": (
            "Heights use one workshop-wide scale calibrated on a single hand-measured pile. "
            "Only shadows confirmed in both raking-light photographs receive a height; "
            "dark base dye visible in the diffuse photograph is never counted as a shadow."
        ),
    }
    overlay = relief_overlay(views[0].frame, height_field, measured_mask, domain, excluded_dark, anchors, factor)
    # Compact height field for later cover-edge integration.
    metrics["height_field"] = encode_height_field(height_field, measured_mask, extent)
    return metrics, overlay


def encode_height_field(height_field, measured_mask, extent):
    """Store measured cells as run-length rows to keep JSON compact."""
    rows = []
    for y in range(height_field.shape[0]):
        active = np.flatnonzero(measured_mask[y])
        if not active.size:
            continue
        splits = np.flatnonzero(np.diff(active) > 1)
        starts = np.r_[active[0], active[splits + 1]]
        stops = np.r_[active[splits], active[-1]] + 1
        for start, stop in zip(starts, stops):
            values = np.round(height_field[y, start:stop], 3)
            # Collapse constant-height plateaus (the usual pile footprint) to
            # one repeated value instead of a number per pixel.
            unique = np.unique(values)
            if unique.size == 1:
                rows.append([int(y), int(start), int(stop), round(float(unique[0]), 3)])
            else:
                rows.append([int(y), int(start), int(stop), values.tolist()])
    return {"width": extent[0], "height": extent[1], "runs": rows}


def decode_height_field(record):
    field = record
    width, height = field["width"], field["height"]
    heights = np.zeros((height, width), dtype=np.float32)
    measured = np.zeros((height, width), dtype=bool)
    for y, start, stop, values in field["runs"]:
        if isinstance(values, (int, float)):
            heights[y, start:stop] = values
        else:
            heights[y, start:stop] = values
        measured[y, start:stop] = True
    return heights, measured


def relief_palette(normalized):
    boundaries = [stop for stop, _ in HEIGHT_STOPS]
    channels = np.asarray([color for _, color in HEIGHT_STOPS], dtype=np.float32) / 255
    result = np.zeros(normalized.shape + (3,), dtype=np.float32)
    for channel in range(3):
        result[..., channel] = np.interp(normalized, boundaries, channels[:, channel])
    return result


def relief_overlay(frame, height_field, measured, domain, excluded_dark, anchors, factor):
    base = frame.display() * 0.55
    peak = float(height_field[measured].max()) if measured.any() else 1.0
    peak = max(peak, 1e-6)
    normalized = np.clip(height_field / peak, 0, 1)
    colors = relief_palette(normalized)
    base[measured] = base[measured] * 0.30 + colors[measured] * 0.70
    unmeasured = domain & ~measured
    yy, xx = np.ogrid[:base.shape[0], :base.shape[1]]
    stripe = ((xx + yy) % 14 <= 3) & unmeasured
    base[unmeasured] = UNMEASURED_TINT
    base[stripe] = UNMEASURED_STRIPE
    base[excluded_dark] = base[excluded_dark] * 0.45 + np.array((0.34, 0.20, 0.24), dtype=np.float32) * 0.55
    image = uint8_image(base, frame.valid.astype(np.float32)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for anchor in anchors:
        points = [(round(x * width), round(y * height)) for x, y in anchor["vertices"]]
        draw.line([*points, points[0]], fill=(240, 214, 120), width=2, joint="curve")
        draw.text((points[0][0] + 4, points[0][1] - 12), anchor["label"][:18], fill=INK,
                  stroke_width=2, stroke_fill=SHADOW)
    return image


def sample_along_polyline(points, extent):
    """Densely sample a cover-edge polyline in registered pixel coordinates."""
    xy = np.asarray(points, dtype=np.float64) * np.asarray(extent) - 0.5
    station_points = [xy[0]]
    segment_starts = [0]
    for start, end in zip(xy[:-1], xy[1:]):
        length = float(np.linalg.norm(end - start))
        count = max(2, int(math.ceil(length)) + 1)
        pieces = np.linspace(start, end, count)[1:]
        segment_starts.append(len(station_points))
        station_points.extend(pieces)
    stations = np.asarray(station_points, dtype=np.float32)
    return stations, xy, segment_starts


def cover_edge_length(store, tile, payload, relief_record):
    """Sloped 3D length of a drawn cover edge following the measured relief.

    Stations whose footprint has no two-view height estimate are reported as
    gaps and excluded from the sloped total instead of being flattened in.
    """
    from .geometry import specimen_frame

    points = cover_points(payload)
    frame = specimen_frame(store.blobs, tile)
    if frame.scale is None:
        raise InputError("Enter physical specimen dimensions before measuring a cover edge.")
    extent = frame.extent
    scale_x, scale_y = frame.scale
    encoded = relief_record["metrics"].get("height_field")
    if not encoded:
        raise InputError("The selected relief measurement does not contain a height field.")
    heights, measured = decode_height_field(encoded)
    if (heights.shape[1], heights.shape[0]) != extent:
        raise InputError("The relief field was built on a different registered plane. Re-run relief after re-registration.")

    stations, xy, segment_starts = sample_along_polyline(points, extent)
    rows = np.clip(np.rint(stations[:, 1]).astype(int), 0, heights.shape[0] - 1)
    cols = np.clip(np.rint(stations[:, 0]).astype(int), 0, heights.shape[1] - 1)
    station_measured = measured[rows, cols] & frame.valid[rows, cols]
    station_heights = heights[rows, cols]

    # Physical 3D positions of every station.
    physical = stations * np.asarray([scale_x, scale_y])
    physical = np.column_stack((physical, station_heights))
    planar_steps = np.linalg.norm(np.diff(physical[:, :2], axis=0), axis=1)
    spatial_steps = np.diff(physical, axis=0)
    step_lengths = np.linalg.norm(spatial_steps, axis=1)
    # A step is usable only when both endpoint stations carry heights.
    usable = station_measured[:-1] & station_measured[1:]
    gap_steps = ~usable

    sloped_total = float(np.sum(step_lengths[usable]))
    projected_total = float(np.sum(planar_steps))
    measured_planar = float(np.sum(planar_steps[usable]))
    gap_planar = float(np.sum(planar_steps[gap_steps]))

    gap_intervals = run_intervals(gap_steps)
    segment_rows = []
    for index in range(len(xy) - 1):
        lo = segment_starts[index]
        hi = segment_starts[index + 1] if index + 1 < len(segment_starts) else len(stations) - 1
        seg_step_hi = max(lo, hi - 1)
        seg_usable = usable[lo:seg_step_hi]
        seg_planar = float(np.sum(planar_steps[lo:seg_step_hi]))
        seg_sloped = float(np.sum(step_lengths[lo:seg_step_hi][seg_usable])) if seg_usable.any() else 0.0
        segment_rows.append({
            "index": index + 1,
            "projected_length_mm": round(seg_planar, 4),
            "sloped_length_mm_measured_only": round(seg_sloped, 4),
            "fully_estimated": bool(seg_usable.all()) if len(seg_usable) else False,
            "stations_with_height": int(seg_usable.sum()),
            "stations_total": int(hi - lo),
        })

    gap_rows = []
    for start_step, stop_step in gap_intervals:
        gap_rows.append({
            "from_station": int(start_step),
            "to_station": int(stop_step + 1),
            "projected_length_mm": round(float(np.sum(planar_steps[start_step:stop_step + 1])), 4),
            "start_normalized": [round(float(stations[start_step, 0] / extent[0]), 5),
                                 round(float(stations[start_step, 1] / extent[1]), 5)],
            "end_normalized": [round(float(stations[stop_step + 1, 0] / extent[0]), 5),
                               round(float(stations[stop_step + 1, 1] / extent[1]), 5)],
        })

    metrics = {
        "relief_measurement_id": relief_record["id"],
        "scale_factor_mm_per_mm_shadow": relief_record["metrics"]["scale_factor_mm_per_mm_shadow"],
        "projected_length_mm": round(projected_total, 4),
        "sloped_length_mm_measured_only": round(sloped_total, 4),
        "measured_projected_portion_mm": round(measured_planar, 4),
        "unestimated_gap_projected_mm": round(gap_planar, 4),
        "station_count": len(stations),
        "stations_with_height": int(np.count_nonzero(station_measured)),
        "stations_without_height": int(np.count_nonzero(~station_measured)),
        "gap_count": len(gap_rows),
        "gaps": gap_rows,
        "segments": segment_rows,
        "note": (
            "Sloped length integrates the real incline between stations with two-view height estimates. "
            "Unestimated gaps are listed and excluded; they are not flattened into the cover total."
        ),
    }
    overlay = cover_overlay(frame, points, stations, station_measured, metrics)
    return metrics, overlay


def run_intervals(flag_array):
    if not flag_array.any():
        return []
    padded = np.pad(flag_array.astype(np.int8), 1)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1
    return list(zip(starts.tolist(), stops.tolist()))


def cover_overlay(frame, points, stations, station_measured, metrics):
    image = base_rgba(frame)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    good = [tuple(point) for point, ok in zip(stations, station_measured) if ok]
    bad = [tuple(point) for point, ok in zip(stations, station_measured) if not ok]
    if len(good) > 1:
        draw.line(good, fill=SHADOW, width=5, joint="curve")
        draw.line(good, fill=MEASURED_TINT, width=3, joint="curve")
    if len(bad) > 1:
        draw.line(bad, fill=SHADOW, width=5, joint="curve")
        draw.line(bad, fill=GAP_TINT, width=3, joint="curve")
    vertices = [(round(x * width), round(y * height)) for x, y in points]
    for index, (x, y) in enumerate(vertices, start=1):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=INK, outline=SHADOW, width=1)
        draw.text((x + 6, y + 2), str(index), fill=INK, stroke_width=2, stroke_fill=SHADOW)
    draw.text((10, 8),
              f"sloped {metrics['sloped_length_mm_measured_only']:.2f} mm of "
              f"projected {metrics['projected_length_mm']:.2f} mm; gaps {metrics['gap_count']}",
              fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image
