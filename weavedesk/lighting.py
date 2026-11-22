import numpy as np

from .color import float_list, quantile_rows
from .defects import gaussian_supported, robust_noise
from .model import InputError
from .photos import luminance


MIN_COMMON_PIXELS = 140
MIN_OVERLAP_FRACTION = 0.12


def captured_pair(primary, comparison, area):
    common = area & primary.valid & comparison.valid
    common_count = int(np.count_nonzero(common))
    available = int(np.count_nonzero(area))
    if common_count < MIN_COMMON_PIXELS or common_count < available * MIN_OVERLAP_FRACTION:
        raise InputError("The photos need a larger common area inside the registered specimen plane.")
    good = common & ~primary.clipped & ~comparison.clipped
    good_count = int(np.count_nonzero(good))
    if good_count < MIN_COMMON_PIXELS:
        raise InputError("Too little shared, unclipped detail remains in the selected area.")
    return common, good, {
        "requested_pixels": available,
        "common_pixels": common_count,
        "unclipped_pixels": good_count,
        "clipped_pairs": common_count - good_count,
        "overlap_percent": round(100 * common_count / available, 2),
        "pair_clipped_percent": round(100 * (common_count - good_count) / common_count, 2),
    }


def registration_pair(primary, comparison):
    if primary.extent != comparison.extent:
        raise InputError("Both measurements require the same registered specimen plane.")
    if primary.photo["id"] == comparison.photo["id"]:
        raise InputError("Choose two different photos.")
    return {
        "primary_id": primary.photo["id"],
        "primary_role": primary.photo["role"],
        "comparison_id": comparison.photo["id"],
        "comparison_role": comparison.photo["role"],
    }


def robust_exposure_ratio(primary, secondary, candidate):
    eligible = candidate & (primary >= 0.025) & (secondary >= 0.025) & (primary <= 0.91) & (secondary <= 0.91)
    ratios = primary[eligible] / secondary[eligible]
    if len(ratios) < 100:
        raise InputError("Too little middle-range shared detail to estimate the exposure ratio.")
    quantiles = np.quantile(ratios, (0.10, 0.25, 0.50, 0.75, 0.90))
    estimate = float(quantiles[2])
    if not 0.2 <= estimate <= 5.0:
        raise InputError("Shared intensity differs by too much to normalize these captures.")
    return estimate, {
        "estimated": True,
        "candidates": len(ratios),
        "ratio_quantiles_10_25_50_75_90": float_list(quantiles, 5),
    }


def fixed_gain(value):
    return value, {"estimated": False, "ratio": value}


def directional_grid(signal, support, active, cells=5):
    height, width = support.shape
    ys = np.rint(np.linspace(0, height, cells + 1)).astype(int)
    xs = np.rint(np.linspace(0, width, cells + 1)).astype(int)
    items = []
    for row, (top, bottom) in enumerate(zip(ys, ys[1:])):
        for col, (left, right) in enumerate(zip(xs, xs[1:])):
            region = np.s_[top:bottom, left:right]
            valid = support[region]
            if np.count_nonzero(valid) < 20:
                continue
            observed = signal[region][valid]
            localized = active[region][valid]
            items.append({
                "cell": [col, row],
                "median_signed": round(float(np.median(observed)), 5),
                "strong_percent": round(100 * np.mean(localized), 2),
                "positive_percent": round(100 * np.mean(observed > 0), 2),
            })
    return items


def measure_direction(first, second, area, parameters):
    pairing = registration_pair(first, second)
    if first.photo["role"] != "rake-left" or second.photo["role"] != "rake-right":
        raise InputError("Direction comparison needs one left and one right raking-light photo.")
    common, fit_area, overlap = captured_pair(first, second, area)
    a = luminance(first.linear)
    b = luminance(second.linear)
    if parameters["balance"] == "auto":
        gain, method = robust_exposure_ratio(a, b, fit_area)
    else:
        gain, method = fixed_gain(parameters["fixed_gain"])
    b = b * gain
    ratio = (a - b) / np.maximum(a + b, 0.045)
    baseline, smooth_coverage = gaussian_supported(ratio, fit_area, parameters["scale_px"])
    residual = ratio - baseline if parameters["remove_broad"] else ratio
    support = fit_area & (smooth_coverage >= 0.68) if parameters["remove_broad"] else fit_area
    if np.count_nonzero(support) < MIN_COMMON_PIXELS:
        raise InputError("The common area is too fragmented for the selected shading scale.")
    values = residual[support]
    floor = robust_noise(values)
    limit = max(parameters["strength_floor"], 2.8 * floor)
    active = support & (np.abs(residual) >= limit)
    result = {
        "pair": pairing,
        "support": overlap,
        "balance": method,
        "gain": round(gain, 5),
        "local_threshold": round(limit, 5),
        "left_strong_percent": round(100 * np.mean(values >= limit), 2),
        "right_strong_percent": round(100 * np.mean(values <= -limit), 2),
        "signed_quantiles": quantile_rows(values),
        "absolute_quantiles": quantile_rows(np.abs(values)),
        "cells": directional_grid(residual, support, active),
        "note": "A registered signed contrast field. Highlights and shadows are relative to these two captures.",
    }
    return result, residual, support, common & ~support
