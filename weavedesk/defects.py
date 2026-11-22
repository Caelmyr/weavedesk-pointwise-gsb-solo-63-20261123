from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from .color import float_list
from .geometry import normalized_bbox
from .model import InputError
from .photos import luminance


CONNECTIVITY = ndimage.generate_binary_structure(2, 2)
MAX_OBSERVATIONS = 160
MAX_COMPONENTS = 5000
MIN_CONFIDENCE_SAMPLES = 36


@dataclass(frozen=True)
class DefectResponse:
    response: np.ndarray
    threshold: float
    valid: np.ndarray
    polarity: str
    noise_floor: float
    local: np.ndarray


def gaussian_supported(values, mask, sigma):
    mask_float = mask.astype(np.float32)
    numerator = ndimage.gaussian_filter(values * mask_float, sigma, mode="constant", cval=0)
    support = ndimage.gaussian_filter(mask_float, sigma, mode="constant", cval=0)
    output = np.divide(numerator, support, out=np.zeros_like(numerator), where=support > 0.001)
    return output, support


def robust_noise(signal):
    median = float(np.median(signal))
    mad = float(np.median(np.abs(signal - median)))
    return max(0.003, mad * 1.4826)


def directional_response(values, mask, radius, sensitivity, polarity):
    broad, broad_support = gaussian_supported(values, mask, radius)
    narrow, narrow_support = gaussian_supported(values, mask, max(0.65, radius * 0.23))
    delta = broad - narrow
    if polarity == "light":
        delta = -delta
    confidence = mask & (broad_support >= 0.77) & (narrow_support >= 0.93)
    signal = delta[confidence]
    if len(signal) < MIN_CONFIDENCE_SAMPLES:
        raise InputError("The selected area has too little continuous detail for this spot size.")
    noise = robust_noise(signal)
    threshold = max(0.011, (5.3 - 3.4 * sensitivity) * noise)
    return DefectResponse(delta, threshold, confidence, polarity, noise, broad)


def detected_islands(response):
    candidates = (response.response >= response.threshold) & response.valid
    labels, count = ndimage.label(candidates, CONNECTIVITY)
    if count > MAX_COMPONENTS:
        raise InputError("This sensitivity produces too many candidates. Reduce it or inspect a smaller region.")
    return labels, ndimage.find_objects(labels)


def marker_outline(mask, bounds, extent):
    coverage = ndimage.binary_erosion(mask, structure=CONNECTIVITY)
    edge = mask & ~coverage
    row, col = np.nonzero(edge)
    if not len(row):
        return []
    cx = np.mean(col)
    cy = np.mean(row)
    angles = np.arctan2(row - cy, col - cx)
    order = np.argsort(angles, kind="stable")
    stride = max(1, int(math.ceil(len(order) / 36)))
    width, height = extent
    y_slice, x_slice = bounds
    return [
        [round(float((col[index] + x_slice.start + 0.5) / width), 5), round(float((row[index] + y_slice.start + 0.5) / height), 5)]
        for index in order[::stride]
    ]


def component_properties(labels, region, label_id, response, frame, area_limit):
    island = labels[region] == label_id
    size = int(np.count_nonzero(island))
    if not area_limit[0] <= size <= area_limit[1]:
        return None
    row, col = np.nonzero(island)
    cy, cx = float(np.mean(row)), float(np.mean(col))
    y_slice, x_slice = region
    height = y_slice.stop - y_slice.start
    width = x_slice.stop - x_slice.start
    edge = island & ~ndimage.binary_erosion(island, structure=CONNECTIVITY)
    edge_size = int(np.count_nonzero(edge))
    peak = float(np.max(response.response[region][island]))
    local_brightness = float(np.median(response.local[region][island]))
    longest = max(width, height)
    narrowest = min(width, height)
    kind = "elongated mark" if longest >= 3 * narrowest and longest >= 5 else "spot"
    extent_width, extent_height = frame.extent
    result = {
        "type": kind,
        "polarity": response.polarity,
        "area_pixels": size,
        "area_mm2": round(size * frame.scale[0] * frame.scale[1], 4) if frame.scale else None,
        "normalized_center": [round((cx + x_slice.start + 0.5) / extent_width, 6), round((cy + y_slice.start + 0.5) / extent_height, 6)],
        "bounds": float_list(normalized_bbox(region, frame.extent), 5),
        "diameter_pixels": round(2 * math.sqrt(size / math.pi), 2),
        "edge_pixels": edge_size,
        "peak_contrast": round(peak, 5),
        "local_lightness": round(local_brightness, 5),
        "strength": round(peak / response.threshold, 3),
        "outline": marker_outline(island, region, frame.extent),
    }
    return result


def candidates_for_response(response, frame, minimum, maximum):
    labels, objects = detected_islands(response)
    found = []
    for index, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        data = component_properties(labels, bounds, index, response, frame, (minimum, maximum))
        if data is not None:
            found.append(data)
    return found


def spatial_density(markers, frame, eligible_area):
    chosen = np.zeros((4, 4), dtype=np.int32)
    for item in markers:
        x, y = item["normalized_center"]
        chosen[min(int(y * 4), 3), min(int(x * 4), 3)] += 1
    count = int(np.count_nonzero(eligible_area))
    measured_area_mm2 = count * frame.scale[0] * frame.scale[1] if frame.scale else None
    return {
        "grid": chosen.tolist(),
        "supported_pixels": count,
        "supported_square_mm": round(measured_area_mm2, 2) if measured_area_mm2 is not None else None,
        "observations_per_square_cm": round(100 * len(markers) / measured_area_mm2, 3) if measured_area_mm2 else None,
    }


def measure_defects(frame, linear, domain, settings):
    area = domain & frame.valid & ~frame.clipped
    values = luminance(linear)
    polarity_settings = ("dark", "light") if settings["polarity"] == "both" else (settings["polarity"],)
    observations = []
    noise = {}
    thresholds = {}
    eligibility = None
    for polarity in polarity_settings:
        response = directional_response(values, area, settings["spot_radius"], settings["sensitivity"], polarity)
        noise[polarity] = round(response.noise_floor, 5)
        thresholds[polarity] = round(response.threshold, 5)
        observed = candidates_for_response(response, frame, settings["minimum_area"], settings["maximum_area"])
        observations.extend(observed)
        eligibility = response.valid if eligibility is None else eligibility & response.valid
    observations.sort(
        key=lambda entry: (-entry["strength"], entry["normalized_center"][1], entry["normalized_center"][0]),
    )
    visible = observations[:MAX_OBSERVATIONS]
    for index, observation in enumerate(visible, 1):
        observation["key"] = f"{observation['polarity'][0].upper()}{index:03d}"
    return {
        "total_observations": len(observations),
        "reported_observations": len(visible),
        "truncated": len(observations) > len(visible),
        "noise_floors": noise,
        "thresholds": thresholds,
        "density": spatial_density(observations, frame, eligibility),
        "observations": visible,
    }
