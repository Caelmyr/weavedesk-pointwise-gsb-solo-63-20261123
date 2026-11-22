import numpy as np
from scipy import ndimage

from .color import delta_e, float_list, linear_lab, lab_linear, swatch
from .geometry import polygon_mask
from .model import InputError


MIN_CHIP_SAMPLES = 40
MIN_SEPARATION = 5.0
MAX_ROWS = 12


def reference_region(tile, purpose, selection):
    candidates = [item for item in tile["regions"] if item["purpose"] == purpose]
    if selection:
        region = next((item for item in candidates if item["id"] == selection), None)
        if region is None:
            raise InputError(f"Choose a saved {purpose} reference region from this specimen.")
        return region
    if not candidates:
        raise InputError(f"Mark a {purpose} reference region before measuring blade coverage.")
    if len(candidates) > 1:
        raise InputError(f"Choose which {purpose} reference region to use.")
    return candidates[0]


def sample_chip(frame, lab, available, region):
    stencil = polygon_mask(frame.extent, region["vertices"])
    support = stencil & available & frame.valid & ~frame.clipped
    count = np.count_nonzero(support)
    if count < MIN_CHIP_SAMPLES:
        raise InputError(f"{region['name']} has too few unclipped reference pixels.")
    values = lab[support]
    anchor = np.median(values, axis=0)
    return anchor, support, {
        "id": region["id"],
        "label": region["name"],
        "lab": float_list(anchor),
        "swatch": swatch(lab_linear(anchor)),
        "pixel_count": int(count),
        "p90_delta_e": round(float(np.quantile(delta_e(values, anchor), 0.90)), 3),
    }


def estimated_fraction(lab, undercoat, coated):
    vector = coated - undercoat
    denominator = float(vector @ vector)
    normalized = np.einsum("...i,i->...", lab - undercoat, vector) / denominator
    clipped = np.clip(normalized, 0, 1)
    reconstructed = undercoat + clipped[:, :, None] * vector
    residual = delta_e(lab, reconstructed)
    return clipped, residual


def classify_fraction(fraction, off_axis, available, *, ambiguity, residual_limit):
    low = 0.5 - ambiguity
    high = 0.5 + ambiguity
    states = np.full(fraction.shape, -1, dtype=np.int8)
    state_ok = available & (off_axis <= residual_limit)
    states[available] = 1
    states[state_ok & (fraction < low)] = 0
    states[state_ok & (fraction > high)] = 2
    return states


def coverage_counts(states, area, scale):
    counts = {
        "bare": int(np.count_nonzero(states == 0)),
        "mixed": int(np.count_nonzero(states == 1)),
        "glazed": int(np.count_nonzero(states == 2)),
        "unmeasured": int(np.count_nonzero(area & (states < 0))),
    }
    total = counts["bare"] + counts["mixed"] + counts["glazed"]
    total_requested = int(np.count_nonzero(area))
    return {
        "pixels": counts,
        "measured": total,
        "percent": {
            key: round(100 * value / (total_requested if key == "unmeasured" else total), 2)
            if total else None
            for key, value in counts.items()
        },
        "area_mm2": {key: round(value * scale[0] * scale[1], 2) for key, value in counts.items()} if scale else None,
    }


def depth_bands(states, area, scale):
    height = states.shape[0]
    edges = np.rint(np.linspace(0, height, MAX_ROWS + 1)).astype(int)
    result = []
    for first, last in zip(edges, edges[1:]):
        band = area[first:last]
        if not band.any():
            continue
        row = coverage_counts(states[first:last], band, scale)
        result.append({
            "relative_top": round(first / height, 4),
            "relative_bottom": round(last / height, 4),
            **row,
        })
    return result


def connected_bare(states, area, minimum_size, scale):
    mask = states == 0
    labels, count = ndimage.label(mask)
    if not count:
        return {"count_above_cutoff": 0, "largest_px": [], "largest_mm2": None if scale is None else []}
    counts = np.bincount(labels.ravel())[1:]
    distances = ndimage.distance_transform_edt(np.pad(area, 1))[1:-1, 1:-1]
    boundary_labels = np.unique(labels[(distances > 0) & (distances <= 1.01)])
    independent = counts.astype(np.int64)
    independent[boundary_labels[boundary_labels > 0] - 1] = 0
    keep = np.sort(independent[independent >= minimum_size])[::-1]
    return {
        "count_above_cutoff": int(len(keep)),
        "largest_px": keep[:8].tolist(),
        "largest_mm2": [round(float(value * scale[0] * scale[1]), 3) for value in keep[:8]] if scale else None,
    }


def measure_coverage(tile, frame, linear, area, unrestricted, parameters):
    lab = linear_lab(linear)
    bare_region = reference_region(tile, "bare", parameters["bare_region"])
    coated_region = reference_region(tile, "glazed", parameters["glazed_region"])
    bare, bare_mask, bare_info = sample_chip(frame, lab, unrestricted, bare_region)
    coated, coated_mask, coated_info = sample_chip(frame, lab, unrestricted, coated_region)
    if np.any(bare_mask & coated_mask):
        raise InputError("The mount and blade reference samples must not overlap.")
    span = float(delta_e(bare, coated))
    if span < MIN_SEPARATION:
        raise InputError("Mount and blade reference colors are too similar for a useful coverage estimate.")
    spread = max(bare_info["p90_delta_e"], coated_info["p90_delta_e"])
    residual_limit = max(4.0, spread * 1.7, span * parameters["max_residual_fraction"])
    scored = frame.valid & ~frame.clipped & area
    if np.count_nonzero(scored) < 36:
        raise InputError("The inspection area contains too few supported pixels to estimate coverage.")
    fraction, residual = estimated_fraction(lab, bare, coated)
    states = classify_fraction(
        fraction,
        residual,
        scored,
        ambiguity=parameters["ambiguity"],
        residual_limit=residual_limit,
    )
    outlying = scored & (residual > residual_limit)
    measurement = {
        **coverage_counts(states, area, frame.scale),
        "bare_reference": bare_info,
        "glazed_reference": coated_info,
        "reference_separation": round(span, 3),
        "off_axis_residual_cutoff": round(residual_limit, 3),
        "off_axis_pixels": int(np.count_nonzero(outlying)),
        "estimated_median_fraction": round(float(np.median(fraction[scored])), 3),
        "depth_bands": depth_bands(states, area, frame.scale),
        "isolated_bare_areas": connected_bare(states, area, parameters["minimum_island"], frame.scale),
    }
    return measurement, states
