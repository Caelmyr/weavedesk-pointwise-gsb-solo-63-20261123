"""Persistent survey direction and segment-by-segment repeat spacing checks.

The rephotography cradle only accepts two perpendicular physical orientations.
A survey plan records which of the two is adopted together with the edge
registration pins and the sampling windows to revisit.  Repeat checks profile
the same physical segments in two registered captures and never substitute a
neighbouring peak where one capture is not resolved.
"""

import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .art import INK, SHADOW, base_rgba, curve_points, font
from .defects import gaussian_supported
from .geometry import polygon_mask, region_domain, specimen_frame, validate_polygon
from .model import InputError, dictionary, numeric_pair, scalar
from .photos import luminance, png_bytes


SURVEY_AXES = (0.0, 90.0)
MIN_REGION_PIXELS = 320
MIN_COHERENCE = 0.12
MIN_SEGMENT_SUPPORT = 0.8
MIN_SEGMENT_COVERAGE = 0.62
MIN_PEAKS = 3
SEGMENTS_CAP = 240
SURVEY_COLOR = (103, 231, 200)
ALT_COLOR = (124, 148, 168)
PIN_COLOR = (248, 207, 108)
OBLIQUE_COLOR = (246, 139, 96)
ALIGNED_COLOR = (103, 231, 200)
BLANK_COLOR = (132, 138, 136)


def axial_difference(first, second):
    return (first - second + 90.0) % 180.0 - 90.0


def survey_parameters(payload):
    data = dictionary(payload, "Survey plan")
    axis = scalar(data.get("survey_axis", SURVEY_AXES[0]), "Adopted survey direction", lower=0, upper=179.999)
    allowed = [round(float(value), 3) for value in SURVEY_AXES]
    if not any(abs(axial_difference(axis, value)) < 0.05 for value in allowed):
        raise InputError("The cradle only accepts its two perpendicular allowed directions.")
    raw_pins = data.get("pins", [])
    if raw_pins:
        pins = [numeric_pair(item, "Registration pin", lower=0, upper=1) for item in raw_pins[:2]]
        if len(pins) != 2:
            raise InputError("Mark both edge registration pins or leave the pin list empty.")
    else:
        pins = []
    return round(axis, 3), pins


def pin_line_angle(pins):
    if len(pins) != 2:
        return None
    delta = np.asarray(pins[1]) - np.asarray(pins[0])
    if float(np.hypot(delta[0], delta[1])) < 0.01:
        raise InputError("The two registration pins need to be visibly separated.")
    return round(math.degrees(math.atan2(delta[1], delta[0])) % 180.0, 2)


def region_grain(gray, mask):
    rows, cols = np.nonzero(mask)
    if len(rows) < MIN_REGION_PIXELS:
        return None
    background, weight = gaussian_supported(gray, mask.astype(np.float32), 5.0)
    trusted = mask & (weight > 0.7)
    if np.count_nonzero(trusted) < MIN_REGION_PIXELS:
        return None
    high = gray - background
    smoothed = ndimage.gaussian_filter(high, 2.2)
    gx = ndimage.sobel(smoothed, axis=1, mode="nearest") / 8
    gy = ndimage.sobel(smoothed, axis=0, mode="nearest") / 8
    ixx = float(np.mean(gx[trusted] ** 2))
    iyy = float(np.mean(gy[trusted] ** 2))
    ixy = float(np.mean(gx[trusted] * gy[trusted]))
    energy = ixx + iyy
    if energy < 1e-10:
        return None
    coherence = min(math.hypot(ixx - iyy, 2 * ixy) / energy, 1.0)
    if coherence < MIN_COHERENCE:
        return {"grain_angle_deg": None, "coherence": round(coherence, 3), "skew_deg": None, "oblique": None}
    gradient_angle = 0.5 * math.degrees(math.atan2(2 * ixy, ixx - iyy))
    grain = (gradient_angle + 90.0) % 180.0
    return {
        "grain_angle_deg": round(grain, 2),
        "coherence": round(coherence, 3),
        "skew_deg": None,
        "oblique": None,
    }


def _line_points(center, angle_deg, length, scale):
    theta = math.radians(angle_deg)
    dx = math.cos(theta) * length / scale[0]
    dy = math.sin(theta) * length / scale[1]
    x, y = center
    return (x - dx, y - dy), (x + dx, y + dy)


def survey_preview(store, tile, payload):
    axis, pins = survey_parameters(payload)
    frame = specimen_frame(store.blobs, tile)
    scale = frame.scale or (1.0, 1.0)
    width, height = frame.extent
    gray = luminance(frame.linear)
    image = base_rgba(frame)
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(min(width, height) / 320))
    center_px = (width / 2.0, height / 2.0)
    reach = min(width * scale[0], height * scale[1]) * 0.42

    region_report = []
    for region in tile["regions"]:
        if region["purpose"] == "ignore":
            continue
        mask = region_domain(frame.extent, tile, region) & frame.valid
        reading = region_grain(gray, mask)
        skew = oblique = None
        if reading and reading["grain_angle_deg"] is not None:
            skew = abs(axial_difference(reading["grain_angle_deg"], axis))
            oblique = skew > 12.0
            reading["skew_deg"] = round(skew, 2)
            reading["oblique"] = bool(oblique)
        points = curve_points(region["vertices"], image.size)
        color = OBLIQUE_COLOR if oblique else (160, 196, 150)
        draw.line([*points, points[0]], fill=SHADOW, width=stroke + 3, joint="curve")
        draw.line([*points, points[0]], fill=color, width=stroke + 1, joint="curve")
        label = region["name"][:30]
        if skew is not None:
            label += f"  {skew:.0f}° skew"
        draw.text((points[0][0] + 4, points[0][1] + 4), label, fill=INK,
                  stroke_width=2, stroke_fill=SHADOW)
        region_report.append({
            "region_id": region["id"],
            "region_name": region["name"],
            **(reading or {"grain_angle_deg": None, "coherence": None, "skew_deg": None, "oblique": None}),
        })

    for index, angle in enumerate(SURVEY_AXES):
        chosen = abs(axial_difference(angle, axis)) < 0.05
        start, end = _line_points(center_px, angle, reach, scale)
        draw.line((start, end), fill=SHADOW, width=(stroke + 4 if chosen else stroke + 2))
        draw.line((start, end), fill=SURVEY_COLOR if chosen else ALT_COLOR,
                  width=(stroke + 2 if chosen else stroke))
        label = f"{angle:.0f}°  {'ADOPTED' if chosen else 'alternative'}"
        draw.text((end[0] + 5, end[1] - 8), label, fill=SURVEY_COLOR if chosen else ALT_COLOR,
                  stroke_width=2, stroke_fill=SHADOW)

    for index, pin in enumerate(pins, 1):
        x, y = pin[0] * width, pin[1] * height
        radius = max(5, round(min(width, height) / 110))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     outline=SHADOW, width=stroke + 2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     outline=PIN_COLOR, width=stroke)
        draw.text((x + radius + 3, y - radius - 2), f"pin {index}", fill=PIN_COLOR,
                  stroke_width=2, stroke_fill=SHADOW)

    caption = f"Adopted survey axis {axis:.0f}° · alternative {(axis + 90) % 180:.0f}°"
    pin_angle = pin_line_angle(pins)
    if pin_angle is not None:
        caption += f" · pin line {pin_angle:.0f}°"
    draw.text((10, 8), caption, fill=INK, stroke_width=2, stroke_fill=SHADOW, font=font(15))

    return {
        "image": image,
        "summary": {
            "survey_axis_deg": axis,
            "allowed_axes_deg": list(SURVEY_AXES),
            "pins": pins,
            "pin_line_angle_deg": pin_angle,
            "regions": region_report,
        },
    }


def save_survey_plan(store, tile_id, payload):
    axis, pins = survey_parameters(payload)
    preview = survey_preview(store, store.tile(tile_id), payload)

    def change(tile):
        if not tile.get("plane"):
            raise InputError("Set a base photo and survey plane before choosing a direction.")
        tile["survey_plan"] = {
            "survey_axis_deg": axis,
            "allowed_axes_deg": list(SURVEY_AXES),
            "pins": pins,
            "pin_line_angle_deg": pin_line_angle(pins),
            "region_ids": [
                region["id"] for region in tile["regions"]
                if region["purpose"] != "ignore"
            ],
            "preview_digest": store.blobs.put(png_bytes(preview["image"])),
        }

    record = store.edit_tile(tile_id, _version(payload), change, "survey",
                             f"Adopted the {axis:.0f}° cradle direction for repeat surveys")
    return record, preview["summary"]


def clear_survey_plan(store, tile_id, payload):
    def change(tile):
        tile.pop("survey_plan", None)

    return store.edit_tile(tile_id, _version(payload), change, "survey",
                           "Cleared the adopted survey direction")


def _version(payload):
    value = payload.get("version")
    if type(value) is not int or value < 1:
        raise InputError("Reload the specimen before saving the survey plan.")
    return value


# ---------------------------------------------------------------------------
# Repeat spacing check
# ---------------------------------------------------------------------------

def repeat_parameters(payload):
    data = dictionary(payload, "Repeat spacing check")
    segment_length = scalar(data.get("segment_length", 42.0), "Segment length (px)", lower=12.0, upper=260.0)
    band_width = scalar(data.get("band_width", 14.0), "Profile band width (px)", lower=3.0, upper=120.0)
    return round(segment_length, 3), round(band_width, 3)


def _axis_unit_vectors(angle_deg):
    """Pixel-space unit vectors along the survey axis and its perpendicular."""
    theta = math.radians(angle_deg)
    along = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    across = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    return along, across


def _physical_length(pixel_vector, scale):
    return float(math.hypot(pixel_vector[0] * scale[0], pixel_vector[1] * scale[1]))


def _profile_coords(origin, along, across, stations, half_band_steps):
    center = origin[None, :] + np.arange(stations)[:, None] * along[None, :]
    if half_band_steps <= 0:
        transverse = np.zeros((1, 2))
    else:
        transverse = (np.arange(-half_band_steps, half_band_steps + 1)[:, None] * across[None, :])
    coords = center[:, None, :] + transverse[None, :, :]
    return coords[..., 1], coords[..., 0]


def _sample_profile(frame, gray, mask, coords):
    rows, cols = coords
    signal = ndimage.map_coordinates(gray, (rows, cols), order=1, mode="constant", cval=0.0)
    support = ndimage.map_coordinates(
        (frame.valid & ~frame.clipped & mask).astype(np.float32),
        (rows, cols), order=1, mode="constant", cval=0.0,
    )
    counts = np.maximum(support.sum(axis=1), 1e-9)
    profile = np.divide((signal * support).sum(axis=1), counts, out=np.zeros_like(counts),
                        where=counts > 0)
    return profile, support.mean(axis=1)


def _detect_pitch(profile, support_fraction, physical_step, expected_range):
    if profile.size < 8 or float(np.mean(support_fraction)) < MIN_SEGMENT_SUPPORT:
        return None, "insufficient image support"
    # A broad Gaussian background removes slow illumination drift without
    # erasing the yarn-period oscillation (window ~ two expected pitches).
    low, high = expected_range
    background_radius = max(3.0, float(np.clip(high / 2.0, 3.0, profile.size / 3.0)))
    background = ndimage.gaussian_filter(profile, background_radius, mode="nearest")
    detrended = profile - background
    # Neighbour differences estimate the noise floor without absorbing the
    # periodic yarn signal itself.
    differences = np.diff(detrended)
    noise = max(float(np.median(np.abs(differences))) / 1.048, 1e-6)
    threshold = 2.0 * noise
    peaks = []
    for index in range(1, profile.size - 1):
        if support_fraction[index] < MIN_SEGMENT_SUPPORT:
            continue
        if detrended[index] >= detrended[index - 1] and detrended[index] >= detrended[index + 1] \
                and detrended[index] >= threshold:
            peaks.append(index)
    if len(peaks) < MIN_PEAKS:
        return None, "fewer than three resolved yarn peaks"
    distances = np.diff(np.asarray(peaks, dtype=np.float64))
    plausible = distances[(distances * physical_step >= low) & (distances * physical_step <= high)]
    if plausible.size < MIN_PEAKS - 1:
        return None, "peak spacing outside the stated yarn range"
    # A handful of merged or split peaks are expected; judge the median spacing
    # on the plausible subset rather than rejecting the whole segment.
    median_spacing = float(np.median(plausible))
    if float(np.median(np.abs(plausible - median_spacing))) / median_spacing > 0.4:
        return None, "irregular peak spacing"
    pitch = median_spacing * physical_step
    return pitch, None


def repeat_spacing(store, tile, plan, photo_a_id, photo_b_id, segment_length, band_width):
    frame_a = specimen_frame(store.blobs, tile, photo_a_id)
    frame_b = specimen_frame(store.blobs, tile, photo_b_id)
    if frame_a.extent != frame_b.extent:
        raise InputError("The two captures no longer share a registered plane; re-register the newer photo.")
    if photo_a_id == photo_b_id:
        raise InputError("Choose the earlier and the later registered capture for the repeat check.")
    width, height = frame_a.extent
    scale = frame_a.scale or (1.0, 1.0)
    axis = float(plan["survey_axis_deg"])
    along, across = _axis_unit_vectors(axis)
    along_physical = _physical_length(along, scale)
    across_physical = _physical_length(across, scale)
    band_steps = max(1, int(round(band_width / max(across_physical, 1e-9))))
    gray_a = luminance(frame_a.linear)
    gray_b = luminance(frame_b.linear)
    unit = "mm" if frame_a.scale else "px"
    # The grid is laid out in pixel space so every segment lands on the same
    # registered pixels in both captures; physical conversion happens on output.
    expected_range_px = (segment_length / 12.0, segment_length / 1.6)

    chosen = [region for region in tile["regions"]
              if region["id"] in plan.get("region_ids", []) and region["purpose"] != "ignore"]
    if not chosen:
        raise InputError("The saved survey plan no longer points at any sampling region.")

    region_rows = []
    segment_rows = []
    overlay = base_rgba(frame_a)
    draw = ImageDraw.Draw(overlay)
    stroke = max(1, round(min(width, height) / 360))

    for region in chosen:
        validate_polygon(region["vertices"])
        mask = polygon_mask(frame_a.extent, region["vertices"])
        rows, cols = np.nonzero(mask & frame_a.valid)
        if len(rows) < MIN_REGION_PIXELS:
            continue
        # Lay the revisit grid over the region's pixel bounding box, then keep
        # only segments whose full length stays inside the marked region.
        row_lo, row_hi = int(rows.min()), int(rows.max()) + 1
        col_lo, col_hi = int(cols.min()), int(cols.max()) + 1
        box_origin = np.array([col_lo + 0.5, row_lo + 0.5], dtype=np.float64)
        box_corner = np.array([col_hi - 0.5, row_hi - 0.5], dtype=np.float64)
        box_points = np.stack((box_origin, box_corner), axis=0)
        longitudinal = (box_points - box_origin) @ np.asarray(along)
        transverse = (box_points - box_origin) @ np.asarray(across)
        stations = max(8, int(math.ceil(segment_length)))
        long_min, long_max = float(longitudinal.min()), float(longitudinal.max())
        trans_min, trans_max = float(transverse.min()), float(transverse.max())
        # Centre each segment one half-window inside the box edges.
        long_offset = (long_max - long_min) % stations / 2.0 + stations / 2.0
        trans_offset = (trans_max - trans_min) % band_width / 2.0 + band_width / 2.0
        long_centers = np.arange(long_min + long_offset, long_max - stations / 2.0, stations)
        trans_centers = np.arange(trans_min + trans_offset, trans_max - band_width / 2.0, band_width)
        segment_index = 0
        region_entries = []

        # Outer loop walks along the adopted survey axis; inner loop lays
        # parallel profiles across it, so every segment runs along the axis.
        for longitudinal_value in long_centers:
            for transverse_value in trans_centers:
                center_px = box_origin + along * longitudinal_value + across * transverse_value
                start_px = center_px - along * (stations - 1) / 2.0
                end_px = center_px + along * (stations - 1) / 2.0
                coords = _profile_coords(start_px, along, across, stations, band_steps)
                coords_flat = (coords[0].ravel(), coords[1].ravel())
                coverage_a = ndimage.map_coordinates(
                    (mask & frame_a.valid & ~frame_a.clipped).astype(np.float32),
                    coords_flat, order=1, mode="constant",
                ).mean()
                coverage_b = ndimage.map_coordinates(
                    (mask & frame_b.valid & ~frame_b.clipped).astype(np.float32),
                    coords_flat, order=1, mode="constant",
                ).mean()
                inside_region = ndimage.map_coordinates(
                    mask.astype(np.float32), coords_flat, order=1, mode="constant", cval=0,
                )
                if float(inside_region.mean()) < MIN_SEGMENT_COVERAGE:
                    continue
                center_col = int(round(center_px[0] - 0.5))
                center_row = int(round(center_px[1] - 0.5))
                if not (0 <= center_col < width and 0 <= center_row < height
                        and frame_a.valid[center_row, center_col]
                        and frame_b.valid[center_row, center_col]):
                    continue
                if coverage_a >= MIN_SEGMENT_COVERAGE and coverage_b >= MIN_SEGMENT_COVERAGE:
                    profile_a, support_a = _sample_profile(frame_a, gray_a, mask, coords)
                    profile_b, support_b = _sample_profile(frame_b, gray_b, mask, coords)
                    pitch_a_px, reason_a = _detect_pitch(profile_a, support_a, 1.0, expected_range_px)
                    pitch_b_px, reason_b = _detect_pitch(profile_b, support_b, 1.0, expected_range_px)
                    if pitch_a_px is None or pitch_b_px is None:
                        # Either capture unresolved: the whole segment stays
                        # blank; the resolved side is never used on its own.
                        pitch_a = pitch_b = None
                        reason = reason_a or reason_b
                    else:
                        pitch_a = pitch_a_px * along_physical
                        pitch_b = pitch_b_px * along_physical
                        reason = None
                else:
                    pitch_a = pitch_b = None
                    reason = "insufficient image support"
                entry = {
                    "region_id": region["id"],
                    "region_name": region["name"],
                    "segment": segment_index + 1,
                    "center_xy": [round(float(center_px[0] / width), 6), round(float(center_px[1] / height), 6)],
                    "start_xy": [round(float(start_px[0] / width), 6), round(float(start_px[1] / height), 6)],
                    "end_xy": [round(float(end_px[0] / width), 6), round(float(end_px[1] / height), 6)],
                    "segment_length": round(stations * along_physical, 4),
                    "support_a_percent": round(float(coverage_a * 100), 2),
                    "support_b_percent": round(float(coverage_b * 100), 2),
                    "pitch_a": round(pitch_a, 4) if pitch_a is not None else None,
                    "pitch_b": round(pitch_b, 4) if pitch_b is not None else None,
                    "pitch_change": round(pitch_b - pitch_a, 4) if pitch_a is not None and pitch_b is not None else None,
                    "pitch_change_percent": round(100 * (pitch_b - pitch_a) / pitch_a, 3)
                    if pitch_a is not None and pitch_b is not None else None,
                    "status": "measured" if pitch_a is not None and pitch_b is not None else "blank",
                    "blank_reason": reason,
                }
                segment_rows.append(entry)
                region_entries.append(entry)
                _draw_segment(draw, start_px, end_px, across, band_width, stroke, entry)
                segment_index += 1
                if len(segment_rows) >= SEGMENTS_CAP:
                    break
            if len(segment_rows) >= SEGMENTS_CAP:
                break
        measured = [item for item in region_entries if item["status"] == "measured"]
        region_rows.append({
            "region_id": region["id"],
            "region_name": region["name"],
            "segments": len(region_entries),
            "measured_segments": len(measured),
            "blank_segments": len(region_entries) - len(measured),
            "median_pitch_a": round(float(np.median([item["pitch_a"] for item in measured])), 4) if measured else None,
            "median_pitch_b": round(float(np.median([item["pitch_b"] for item in measured])), 4) if measured else None,
            "median_change_percent": round(float(np.median(
                [item["pitch_change_percent"] for item in measured])), 3) if measured else None,
        })
        if len(segment_rows) >= SEGMENTS_CAP:
            break

    all_measured = [item for item in segment_rows if item["status"] == "measured"]
    _draw_repeat_legend(draw, width, height)
    metrics = {
        "mode": "repeat",
        "survey_axis_deg": axis,
        "allowed_axes_deg": list(SURVEY_AXES),
        "segment_length": round(segment_length, 4),
        "band_width": round(band_width, 4),
        "unit": unit,
        "photo_a": _photo_brief(frame_a.photo),
        "photo_b": _photo_brief(frame_b.photo),
        "region_rows": region_rows,
        "segments": segment_rows,
        "measured_segments": len(all_measured),
        "blank_segments": len(segment_rows) - len(all_measured),
        "median_pitch_a": round(float(np.median([item["pitch_a"] for item in all_measured])), 4)
        if all_measured else None,
        "median_pitch_b": round(float(np.median([item["pitch_b"] for item in all_measured])), 4)
        if all_measured else None,
        "median_change_percent": round(float(np.median(
            [item["pitch_change_percent"] for item in all_measured])), 3) if all_measured else None,
        "note": "Segments revisit the same physical stations on the adopted survey axis. "
                "A segment stays blank where either capture is unresolved; neighbouring peaks are never substituted.",
    }
    return metrics, overlay


def _photo_brief(photo):
    return {"id": photo["id"], "name": photo["name"], "role": photo["role"]}


def _draw_repeat_legend(draw, width, height):
    entries = (
        (ALIGNED_COLOR, "measured · change within 8%"),
        (OBLIQUE_COLOR, "measured · change above 8%"),
        (BLANK_COLOR, "blank · unresolved in at least one capture"),
    )
    x, y = 12, height - 14 - len(entries) * 20
    draw.rectangle((x - 8, y - 8, x + 330, y + len(entries) * 20 + 4), fill=(24, 31, 34, 210))
    for index, (color, label) in enumerate(entries):
        top = y + index * 20
        draw.rectangle((x, top, x + 18, top + 12), fill=(*color, 200), outline=INK)
        draw.text((x + 26, top - 2), label, fill=INK)


def _draw_segment(draw, start_px, end_px, across, band_width, stroke, entry):
    half = across * band_width / 2.0
    polygon = [tuple(float(value) for value in point)
               for point in (start_px - half, start_px + half, end_px + half, end_px - half)]
    if entry["status"] == "measured":
        color = ALIGNED_COLOR if abs(entry["pitch_change_percent"] or 0.0) < 8.0 else OBLIQUE_COLOR
        fill = (*color, 70)
    else:
        color = BLANK_COLOR
        fill = (*color, 42)
    draw.polygon(polygon, fill=fill)
    draw.line([*polygon, polygon[0]], fill=color, width=stroke)
