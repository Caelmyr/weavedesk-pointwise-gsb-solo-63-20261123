import math

import numpy as np
from scipy import ndimage

from .color import quantile_rows
from .defects import gaussian_supported, robust_noise
from .model import InputError
from .photos import luminance


DIRECTIONS = (
    ("horizontal", 0, 1, 0),
    ("rising diagonal", 1, -1, 45),
    ("vertical", 1, 0, 90),
    ("falling diagonal", 1, 1, 135),
)
MIN_TEXTURE_PIXELS = 100


def shift_overlap(array, dy, dx):
    rows, columns = array.shape
    if abs(dy) >= rows or abs(dx) >= columns:
        raise InputError("Texture spacing exceeds the registered image size.")
    if dy >= 0:
        a_rows = slice(0, rows - dy)
        b_rows = slice(dy, rows)
    else:
        a_rows = slice(-dy, rows)
        b_rows = slice(0, rows + dy)
    if dx >= 0:
        a_cols = slice(0, columns - dx)
        b_cols = slice(dx, columns)
    else:
        a_cols = slice(-dx, columns)
        b_cols = slice(0, columns + dx)
    return array[a_rows, a_cols], array[b_rows, b_cols]


def autocorrelation(signal, valid, direction, spacings):
    name, uy, ux, angle = direction
    table = []
    for step in spacings:
        dy, dx = uy * step, ux * step
        if abs(dy) >= signal.shape[0] or abs(dx) >= signal.shape[1]:
            continue
        a_valid, b_valid = shift_overlap(valid, dy, dx)
        common = a_valid & b_valid
        count = int(np.count_nonzero(common))
        if count < MIN_TEXTURE_PIXELS:
            continue
        whole_a, whole_b = shift_overlap(signal, dy, dx)
        a, b = whole_a[common], whole_b[common]
        a, b = a - a.mean(), b - b.mean()
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        result = float(np.dot(a, b) / norm) if norm > 1e-12 else None
        table.append({
            "step": step,
            "spacing_px": round(math.hypot(dy, dx), 3),
            "sample_pairs": count,
            "correlation": round(result, 5) if result is not None else None,
        })
    return {"direction": name, "angle_degrees": angle, "points": table}


def decorated_periods(curve, scale, direction):
    _, uy, ux, _ = direction
    if scale:
        for point in curve["points"]:
            step = point["step"]
            point["spacing_mm"] = round(math.hypot(scale[0] * ux, scale[1] * uy) * step, 4)
    peaks = []
    for before, point, after in zip(curve["points"], curve["points"][1:], curve["points"][2:]):
        scores = [entry["correlation"] for entry in (before, point, after)]
        if any(score is None for score in scores):
            continue
        if point["step"] != before["step"] + 1 or after["step"] != point["step"] + 1:
            continue
        if scores[1] > max(scores[0], scores[2], 0.15):
            peaks.append(point)
    curve["local_highs"] = sorted(peaks, key=lambda item: -item["correlation"])[:3]
    return curve


def lag_curves(signal, support, requested_max, scale):
    max_step = min(requested_max, min(signal.shape) // 3)
    diagonal = math.ceil(max_step / math.sqrt(2))
    curves = []
    for direction in DIRECTIONS:
        spacing = diagonal if direction[1] and direction[2] else max_step
        curve = autocorrelation(signal, support, direction, range(1, spacing + 1))
        curves.append(decorated_periods(curve, scale, direction))
    return curves


def gradient_pairs(field, mask, sigma):
    fill, _ = gaussian_supported(field, mask, sigma)
    core = ndimage.binary_erosion(mask, iterations=max(1, round(sigma)))
    dx = ndimage.sobel(fill, axis=1, mode="nearest") / 8
    dy = ndimage.sobel(fill, axis=0, mode="nearest") / 8
    return dx, dy, core


def orientation_strength(dx, dy, core):
    if np.count_nonzero(core) < MIN_TEXTURE_PIXELS:
        return None
    gx = dx[core]
    gy = dy[core]
    ixx = float(np.mean(gx * gx))
    iyy = float(np.mean(gy * gy))
    ixy = float(np.mean(gx * gy))
    scale = ixx + iyy
    if scale < 1e-8:
        return None
    angle = math.degrees(0.5 * math.atan2(2 * ixy, ixx - iyy))
    strength = min(math.hypot(ixx - iyy, 2 * ixy) / scale, 1)
    return {
        "ridge_angle_clockwise": round((angle + 90) % 180, 2),
        "coherence": round(strength, 4),
        "gradient_energy": round(scale, 7),
    }


def gradient_histogram(dx, dy, core):
    angle = (np.arctan2(dy[core], dx[core]) + np.pi / 2) % np.pi
    strength = np.hypot(dx[core], dy[core])
    hist, bounds = np.histogram(angle, bins=12, range=(0, np.pi), weights=strength)
    norm = max(float(np.sum(hist)), 1e-8)
    return [
        {
            "start_deg": round(math.degrees(bounds[index])),
            "end_deg": round(math.degrees(bounds[index + 1])),
            "fraction": round(float(value) / norm, 4),
        }
        for index, value in enumerate(hist)
    ]


def multiscale_detail(values, supported, sigma):
    scales = (1.0, max(2.0, sigma / 2), sigma)
    rows = []
    for scale in scales:
        mean, weight = gaussian_supported(values, supported, scale)
        core = supported & (weight >= 0.77)
        residual = values[core] - mean[core]
        if len(residual) < MIN_TEXTURE_PIXELS:
            continue
        rows.append({
            "radius_px": scale,
            "samples": len(residual),
            "roughness_rms": round(float(np.sqrt(np.mean(residual**2))), 5),
            "robust_range": quantile_rows(np.abs(residual), (0.50, 0.90, 0.95), 5),
        })
    return rows


def local_rms(signal, support, radius):
    squared, fraction = gaussian_supported(signal**2, support, radius)
    quality = support & (fraction >= 0.70)
    return np.sqrt(squared), quality


def measure_texture(frame, rgb_linear, domain, parameters):
    valid = frame.valid & ~frame.clipped & domain
    values = luminance(rgb_linear)
    radius = parameters["background_radius"]
    background, weight = gaussian_supported(values, valid, radius)
    support = valid & (weight >= 0.78)
    if np.count_nonzero(support) < MIN_TEXTURE_PIXELS:
        raise InputError("The region needs a wider uninterrupted area at this texture scale.")
    high = values - background
    dx, dy, core = gradient_pairs(high, support, parameters["edge_smooth"])
    amplitudes, display_support = local_rms(high, support, max(radius * 0.45, 1))
    result = {
        "supported_pixels": int(np.count_nonzero(support)),
        "grid_roughness": quantile_rows(amplitudes[display_support], (0.10, 0.50, 0.90, 0.99), 6),
        "detail_noise": round(robust_noise(high[support]), 6),
        "texture_orientation": orientation_strength(dx, dy, core),
        "orientation_histogram": gradient_histogram(dx, dy, core),
        "lag_curves": lag_curves(high, support, parameters["maximum_lag"], frame.scale),
        "scales": multiscale_detail(values, valid, radius),
        "note": "Texture comes from visible image contrast under the selected capture and analysis scale.",
    }
    return result, amplitudes, display_support
