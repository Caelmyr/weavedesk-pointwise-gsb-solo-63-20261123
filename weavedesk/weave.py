"""Measure directional yarn spacing from supported image texture."""

import math

import numpy as np
from PIL import Image, ImageDraw

from .art import base_rgba
from .defects import gaussian_supported
from .model import InputError
from .photos import luminance


def axial_difference(first, second):
    return (first - second + 90) % 180 - 90


def spectrum(signal, support, scale):
    height, width = signal.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    transformed = np.fft.fftshift(np.fft.fft2(signal * window * support))
    fx = np.fft.fftshift(np.fft.fftfreq(width, scale[0]))
    fy = np.fft.fftshift(np.fft.fftfreq(height, scale[1]))
    xx, yy = np.meshgrid(fx, fy)
    frequency = np.hypot(xx, yy)
    angle = (np.degrees(np.arctan2(yy, xx)) + 90) % 180
    return np.abs(transformed) ** 2, frequency, angle


def peak_in_sector(power, frequencies, angles, direction, period_range, tolerance):
    low, high = period_range
    sector = ((frequencies >= 1 / high) & (frequencies <= 1 / low)
              & (np.abs(axial_difference(angles, direction)) <= tolerance))
    readings = power[sector]
    if readings.size < 8:
        raise InputError("The supported area is too small to resolve a peak in the selected spacing range.")
    peak_index = np.unravel_index(np.argmax(np.where(sector, power, -1)), power.shape)
    noise = float(np.median(readings))
    ratio = float(power[peak_index]) / max(noise, 1e-9)
    if ratio < 7 or power[peak_index] <= max(float(readings.sum()) * .012, 1e-9):
        return None
    return {
        "angle_degrees": round(float(angles[peak_index]), 2),
        "pitch": round(1 / float(frequencies[peak_index]), 4),
        "peak_to_background": round(ratio, 2),
    }


def observation_picture(frame, region, support, readings, center, span):
    overlay = base_rgba(frame)
    layer = np.array(overlay)
    mask = region & ~support
    layer[mask, :3] = (layer[mask, :3] * .63 + np.array([227, 156, 91]) * .37).astype(np.uint8)
    image = Image.fromarray(layer)
    draw = ImageDraw.Draw(image)
    sx, sy = frame.scale or (1.0, 1.0)
    for reading, color in zip(readings, ((103, 231, 200), (248, 207, 108))):
        if not reading:
            continue
        theta = math.radians(reading["angle_degrees"])
        delta = (math.cos(theta) * span / sx, math.sin(theta) * span / sy)
        first = tuple(center[i] - delta[i] for i in range(2))
        last = tuple(center[i] + delta[i] for i in range(2))
        draw.line((first, last), fill=(16, 31, 29), width=6)
        draw.line((first, last), fill=color, width=3)
    return image


def measure_weave(frame, linear, region, settings):
    visible = region & frame.valid & ~frame.clipped
    supported_rows, supported_cols = np.nonzero(visible)
    if len(supported_rows) < 400:
        raise InputError("Choose a larger unobscured fabric region.")
    ys = slice(int(supported_rows.min()), int(supported_rows.max()) + 1)
    xs = slice(int(supported_cols.min()), int(supported_cols.max()) + 1)
    support = visible[ys, xs]
    height, width = support.shape
    if min(height, width) < 32 or support.mean() < .65:
        raise InputError("This fabric window needs at least 32 pixels per side and 65% image support.")
    min_px, max_px = settings["minimum_pitch"], settings["maximum_pitch"]
    if min_px >= max_px:
        raise InputError("Maximum pitch must exceed minimum pitch.")
    scale = tuple(frame.scale or (1.0, 1.0))
    lower, upper = min_px * min(scale), max_px * max(scale)
    gray = luminance(linear[ys, xs])
    background, weight = gaussian_supported(gray, support, max(3.0, max_px / 2))
    trusted = support & (weight > .63)
    if trusted.mean() < .55:
        raise InputError("The selected window has too much incomplete support for a weave reading.")
    power, frequencies, angles = spectrum(gray - background, trusted, scale)
    reference = settings["reference_axis"]
    candidates = [
        peak_in_sector(power, frequencies, angles, direction, (lower, upper), settings["tolerance"])
        for direction in (reference, (reference + 90) % 180)
    ]
    first, second = candidates
    crossing = abs(float(axial_difference(first["angle_degrees"], second["angle_degrees"]))) if first and second else None
    unit = "mm" if frame.scale else "px"
    rows = [{"family": family, **(row or {"angle_degrees": None, "pitch": None, "peak_to_background": None})}
            for family, row in zip(("A", "B"), candidates)]
    full_support = np.zeros_like(visible)
    full_support[ys, xs] = trusted
    center = (float(np.median(supported_cols)), float(np.median(supported_rows)))
    span = min(width * scale[0], height * scale[1]) * .19
    return {
        "rows": rows,
        "unit": unit,
        "axis_a_pitch": first["pitch"] if first else None,
        "axis_b_pitch": second["pitch"] if second else None,
        "crossing_angle": round(crossing, 2) if crossing is not None else None,
        "supported_pixels": int(np.count_nonzero(trusted)),
        "note": "The two sectors follow the indicated thread directions. Missing peaks are left blank. Colored axes indicate the supported local directions, not individual thread centerlines.",
    }, observation_picture(frame, region, full_support, candidates, center, span)
