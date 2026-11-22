"""One adopted polarized-light line, fitted from two confirmed example patches.

A specimen registers a single glare line. Polarized runs and reference re-checks
on that specimen all read light behavior through this same line, so the two ends
of one specimen can never be tuned against different lines.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .color import float_list, linear_lab, quantile_rows
from .geometry import polygon_mask, region_domain
from .lighting import captured_pair, registration_pair
from .model import InputError, collection, dictionary, identity
from .photos import luminance
from .store import make_id, timestamp


MIN_EXAMPLE_PIXELS = 40
MAX_EXAMPLES = 8
FIT_FRACTION = 0.02
DETECT_FRACTION = 0.06
MIN_LIGHT = 0.012
RIDGE = 1e-4
LINE_VERSION = 1
CONNECTIVITY = ndimage.generate_binary_structure(2, 2)


@dataclass(frozen=True)
class PairLight:
    fraction: np.ndarray
    loss: np.ndarray
    measurable: np.ndarray
    support: np.ndarray
    gain: np.ndarray
    gain_summary: dict
    overlap: dict


@dataclass(frozen=True)
class SessionStates:
    retained: np.ndarray
    unreliable: np.ndarray
    too_little_detail: np.ndarray
    not_visible: np.ndarray
    score: np.ndarray


def saturation(linear):
    lab = linear_lab(linear)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    return chroma / np.maximum(lab[..., 0], 4.0)


def channel_gain(primary, secondary, area):
    first = luminance(primary)
    second = luminance(secondary)
    active = area & (first >= 0.02) & (second >= 0.02) & (first <= 0.93) & (second <= 0.93)
    if np.count_nonzero(active) < 100:
        raise InputError("Too little middle-range shared detail to balance this polarized pair.")
    ratios = primary[active] / np.maximum(secondary[active], 1e-4)
    gains = np.quantile(ratios, 0.25, axis=0).astype(np.float32)
    if np.any(gains < 0.2) or np.any(gains > 5.0):
        raise InputError("These polarized captures need a more consistent exposure before a glare line can be adopted.")
    return gains, {
        "per_channel": float_list(gains, 5),
        "candidates": int(np.count_nonzero(active)),
        "method": "Lower-quartile channel ratio outside bright specular detail.",
    }


def pair_light(parallel, cross, area):
    if parallel.extent != cross.extent:
        raise InputError("Both polarized captures must share the registered specimen plane.")
    common, good, overlap = captured_pair(parallel, cross, area)
    gains, summary = channel_gain(parallel.linear, cross.linear, good)
    direct = luminance(parallel.linear)
    fitted = luminance(cross.linear * gains)
    fraction = np.maximum(direct - fitted, 0) / np.maximum(direct, 0.025)
    loss = saturation(cross.linear) - saturation(parallel.linear)
    measurable = good & (direct >= MIN_LIGHT) & (fitted >= MIN_LIGHT) & (fraction >= FIT_FRACTION)
    return PairLight(fraction, loss, measurable, good, gains, summary, overlap), common


def robust_scale(values):
    center = float(np.median(values))
    spread = float(np.median(np.abs(values - center))) * 1.4826
    return center, max(spread, 1e-6)


def example_fields(payload):
    data = dictionary(payload, "Glare line examples")
    examples = collection(data.get("examples"), "Glare line examples", minimum=2, maximum=MAX_EXAMPLES)
    result = []
    seen = set()
    for entry in examples:
        item = dictionary(entry, "Glare line example")
        kind = item.get("kind")
        if kind not in ("glare", "color"):
            raise InputError("Each example must be marked as film glare or retained yarn color.")
        region_id = identity(item.get("region_id"), "Glare line example region")
        if region_id in seen:
            raise InputError("Each glare line example needs its own marked region.")
        seen.add(region_id)
        result.append({"kind": kind, "region_id": region_id})
    for required, label in (("glare", "film glare"), ("color", "retained yarn color")):
        if not any(item["kind"] == required for item in result):
            raise InputError(f"Mark at least one region where the {label} is confirmed.")
    return result


def example_region(tile, region_id):
    region = next((item for item in tile["regions"] if item["id"] == region_id), None)
    if region is None:
        raise InputError("A selected glare line example region has been removed.")
    if region["purpose"] == "ignore":
        raise InputError("An exclusion region cannot define the glare line.")
    return region


def example_masks(tile, frame, examples):
    glare = np.zeros(frame.valid.shape, dtype=bool)
    color = np.zeros(frame.valid.shape, dtype=bool)
    summary = []
    for example in examples:
        region = example_region(tile, example["region_id"])
        mask = polygon_mask(frame.extent, region["vertices"]) & frame.valid & ~frame.clipped
        pixels = int(np.count_nonzero(mask))
        if pixels < MIN_EXAMPLE_PIXELS:
            raise InputError(f"{region['name']} has too few supported pixels to define the glare line.")
        if np.any(mask & glare) or np.any(mask & color):
            raise InputError("Glare line example regions must not overlap each other.")
        if example["kind"] == "glare":
            glare |= mask
        else:
            color |= mask
        summary.append({
            "kind": example["kind"],
            "region_id": region["id"],
            "region_name": region["name"],
            "purpose": region["purpose"],
            "vertices": region["vertices"],
            "pixels": pixels,
        })
    return glare, color, summary


def fit_line(light, glare_mask, color_mask):
    usable = light.measurable & (glare_mask | color_mask)
    glare = usable & glare_mask
    color = usable & color_mask
    if np.count_nonzero(glare) < MIN_EXAMPLE_PIXELS or np.count_nonzero(color) < MIN_EXAMPLE_PIXELS:
        raise InputError("The example patches need more measurable polarized detail to define one line.")
    fraction_center, fraction_scale = robust_scale(light.fraction[glare | color])
    loss_center, loss_scale = robust_scale(light.loss[glare | color])
    fraction_scale = max(fraction_scale, 0.04)
    loss_scale = max(loss_scale, 0.015)
    points = np.stack(
        ((light.fraction - fraction_center) / fraction_scale, (light.loss - loss_center) / loss_scale),
        axis=-1,
    )
    samples = np.r_[points[glare], points[color]]
    glare_count = int(np.count_nonzero(glare))
    glare_mean = samples[:glare_count].mean(axis=0)
    color_mean = samples[glare_count:].mean(axis=0)
    difference = color_mean - glare_mean
    if np.linalg.norm(difference) < 0.35:
        raise InputError("The two examples do not separate on light behavior. Mark clearer film glare and clearer yarn color.")
    covariance = np.cov(samples, rowvar=False) + np.eye(2) * RIDGE
    weights = np.linalg.solve(covariance, difference)
    midpoint = (glare_mean + color_mean) / 2
    offset = float(-weights @ midpoint)
    values = points @ weights + offset
    glare_retained = float(np.mean(values[glare] >= 0))
    color_retained = float(np.mean(values[color] >= 0))
    if glare_retained > 0.4 or color_retained < 0.6:
        raise InputError("The two examples overlap too much for one reliable line. Mark more distinct patches.")
    return {
        "fraction_center": round(fraction_center, 6),
        "fraction_scale": round(fraction_scale, 6),
        "loss_center": round(loss_center, 6),
        "loss_scale": round(loss_scale, 6),
        "fraction_weight": round(float(weights[0]), 6),
        "loss_weight": round(float(weights[1]), 6),
        "offset": round(offset, 6),
        "rule": "retained when fraction_weight x scaled light loss + loss_weight x scaled saturation loss + offset >= 0",
    }


def line_score(light, line):
    fraction = (light.fraction - line["fraction_center"]) / line["fraction_scale"]
    loss = (light.loss - line["loss_center"]) / line["loss_scale"]
    return fraction * line["fraction_weight"] + loss * line["loss_weight"] + line["offset"]


def session_states(light, line):
    score = line_score(light, line)
    detail = light.support & ~light.measurable
    return SessionStates(
        retained=light.measurable & (score >= 0),
        unreliable=light.measurable & (score < 0),
        too_little_detail=detail,
        not_visible=~light.support,
        score=score,
    )


def line_signature(line):
    keys = ("fraction_center", "fraction_scale", "loss_center", "loss_scale", "fraction_weight", "loss_weight", "offset")
    return tuple(line.get(key) for key in keys)


def same_line(first, second):
    return line_signature(first) == line_signature(second)


def fit_adopted_line(tile, parallel, cross, examples):
    area = region_domain(parallel.extent, tile, None)
    light, _ = pair_light(parallel, cross, area)
    glare_mask, color_mask, summary = example_masks(tile, parallel, examples)
    line = fit_line(light, glare_mask, color_mask)
    states = session_states(light, line)
    return {
        "id": make_id(),
        "version": LINE_VERSION,
        "created_at": timestamp(),
        "pair": registration_pair(parallel, cross),
        "examples": summary,
        "line": line,
        "balance": light.gain_summary,
        "support": light.overlap,
        "counts": {
            "glare_example_pixels": int(np.count_nonzero(glare_mask)),
            "color_example_pixels": int(np.count_nonzero(color_mask)),
            "example_glare_retained_percent": round(100 * float(np.mean(states.retained[glare_mask])), 2),
            "example_color_retained_percent": round(100 * float(np.mean(states.retained[color_mask])), 2),
        },
        "plane": list(parallel.extent),
        "note": "The specimen's single adopted glare line. Every polarized run and reference re-check reads this same line.",
    }


def current_line(tile):
    line = tile.get("glare_line")
    if not line:
        raise InputError("Adopt this specimen's one polarized glare line before this measurement.")
    return line


def line_reference(record):
    if record is None:
        return None
    return {
        "id": record["id"],
        "version": record["version"],
        "created_at": record["created_at"],
        "pair": record["pair"],
        "line": record["line"],
        "counts": record["counts"],
    }


def components(mask, minimum):
    labels, count = ndimage.label(mask, CONNECTIVITY)
    if not count:
        return []
    height, width = mask.shape
    sizes = np.bincount(labels.ravel())
    result = []
    for index, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        rows, cols = region
        size = int(sizes[index])
        if size < minimum:
            continue
        result.append({
            "pixels": size,
            "bounds": [cols.start / width, rows.start / height, cols.stop / width, rows.stop / height],
        })
    return result


def measure_polarized(parallel, cross, area, record, parameters):
    if parallel.photo["role"] != "parallel" or cross.photo["role"] != "cross":
        raise InputError("Polarized glare needs one parallel-polarized and one cross-polarized capture.")
    line = record["line"]
    light, common = pair_light(parallel, cross, area)
    states = session_states(light, line)
    domain = common & area
    total = max(int(np.count_nonzero(domain)), 1)
    retained = states.retained & domain
    unreliable = states.unreliable & domain
    detail = states.too_little_detail & domain
    hidden = domain & ~light.support
    hotspot_candidates = unreliable & (light.fraction >= parameters["glare_fraction"])
    entries = components(hotspot_candidates, parameters["minimum_hotspot"])
    entries.sort(key=lambda item: -item["pixels"])
    for entry in entries:
        entry["area_mm2"] = round(entry["pixels"] * parallel.scale[0] * parallel.scale[1], 3) if parallel.scale else None
    retained_count = int(np.count_nonzero(retained))
    metrics = {
        "pair": registration_pair(parallel, cross),
        "line": line_reference(record),
        "support": light.overlap,
        "balance": light.gain_summary,
        "retained_pixels": retained_count,
        "retained_percent": round(100 * retained_count / total, 2),
        "unreliable_pixels": int(np.count_nonzero(unreliable)),
        "unreliable_percent": round(100 * np.count_nonzero(unreliable) / total, 2),
        "too_little_detail_percent": round(100 * np.count_nonzero(detail) / total, 2),
        "not_visible_percent": round(100 * np.count_nonzero(hidden) / total, 2),
        "glare_fraction_threshold": parameters["glare_fraction"],
        "retained_light_loss_quantiles": quantile_rows(light.fraction[retained]) if retained_count else None,
        "retained_saturation_quantiles": quantile_rows(saturation(parallel.linear)[retained]) if retained_count else None,
        "hotspots": entries[:24],
        "hotspot_count": len(entries),
        "note": "Color remains only where the specimen's one adopted glare line admits it. Marked bright film glare is kept out of color work.",
    }
    return metrics, states, domain
