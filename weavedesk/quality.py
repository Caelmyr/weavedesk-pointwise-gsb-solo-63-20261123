from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import font
from .color import quantile_rows
from .defects import gaussian_supported
from .geometry import active_region, region_domain, specimen_frame
from .inspection import focus_diagnostics
from .model import dictionary, enumeration, identity, integer
from .photos import luminance, uint8_image


CAPTURE_PALETTE = {
    "edge_clipping": (220, 131, 95),
    "local_detail": (175, 214, 151),
    "source_support": (121, 187, 174),
}
STATES = {
    "ok": (100, 168, 144),
    "dark": (69, 98, 151),
    "bright": (233, 147, 106),
    "absent": (88, 87, 89),
}
MIN_PHOTO_CODE = 5 / 255
MAX_PHOTO_CODE = 253 / 255


def exposure_records(tile):
    seen = Counter(item["role"] for item in tile["photos"])
    result = []
    for photo in tile["photos"]:
        metadata = photo["exif"]
        item = {
            "capture_id": photo["id"],
            "role": photo["role"],
            "name": photo["name"],
            "f_number": metadata.get("f_number"),
            "exposure_seconds": metadata.get("exposure_seconds"),
            "iso": metadata.get("iso"),
            "repeat_role": seen[photo["role"]] > 1,
            "usable_alpha": photo["stats"]["supported_percent"],
            "thumbnail_white_clip": photo["stats"]["white_clip_percent"],
            "thumbnail_black_clip": photo["stats"]["black_clip_percent"],
            "exposure_index": relative_light_index(metadata),
        }
        result.append(item)
    return result


def pair_exposure_span(records):
    values = [row["exposure_index"] for row in records if row["exposure_index"] is not None]
    if len(values) < 2:
        return {"recorded_capture_count": len(values), "span_ev": None}
    low = min(values)
    high = max(values)
    return {
        "recorded_capture_count": len(values),
        "span_ev": round(high - low, 3),
        "lowest_capture_ids": [entry["capture_id"] for entry in records if entry["exposure_index"] == low],
        "highest_capture_ids": [entry["capture_id"] for entry in records if entry["exposure_index"] == high],
    }


def relative_light_index(metadata):
    shutter = metadata.get("exposure_seconds")
    aperture = metadata.get("f_number")
    sensitivity = metadata.get("iso")
    if any(type(value) not in (float, int) or value <= 0 for value in (shutter, aperture, sensitivity)):
        return None
    quantity = shutter * sensitivity / (100 * aperture * aperture)
    return round(float(np.log2(quantity)), 5)


def brightness_class(frame, area):
    displayed = frame.display()
    channels = np.max(displayed, axis=2)
    bright = area & frame.valid & (channels >= MAX_PHOTO_CODE)
    dark = area & frame.valid & (channels <= MIN_PHOTO_CODE)
    missing = area & ~frame.valid
    return bright, dark, missing


def inspected_detail(frame, area):
    gray = luminance(frame.linear)
    support = frame.valid & area
    blurred, weight = gaussian_supported(gray, support, 1.8)
    detail = np.abs(gray - blurred)
    reliable = support & (weight > 0.9) & ~frame.clipped
    values = detail[reliable]
    if not len(values):
        return detail * 0, reliable, 0.0
    error = float(np.quantile(values, 0.95))
    return detail, reliable, error


def fragment_rows(fault, shape, minimum):
    labels, count = ndimage.label(fault)
    if not count:
        return []
    ids = np.argsort(np.bincount(labels.ravel())[1:], kind="stable")[::-1]
    regions = ndimage.find_objects(labels)
    height, width = shape
    result = []
    for index in ids[:24]:
        yy, xx = regions[index]
        hits = int(np.count_nonzero(labels[yy, xx] == index + 1))
        if hits < minimum:
            continue
        result.append({
            "pixels": hits,
            "bounds": [xx.start / width, yy.start / height, xx.stop / width, yy.stop / height],
            "relative_center": [(xx.start + xx.stop) / (2 * width), (yy.start + yy.stop) / (2 * height)],
        })
    return result


def channel_statistics(rgb, support):
    supported = rgb[support]
    if not len(supported):
        return {"rgb_quantiles": None, "per_channel_highlight_clip": None}
    return {
        "rgb_quantiles": [quantile_rows(plane, (0.01, 0.25, 0.50, 0.75, 0.99), 5) for plane in supported.T],
        "per_channel_highlight_clip": [round(100 * np.mean(channel >= MAX_PHOTO_CODE), 3) for channel in supported.T],
    }


def quantity_rows(frame, area, bright, dark, missing):
    req = max(int(np.count_nonzero(area)), 1)
    return {
        "requested_pixels": req,
        "placed_pixels": int(np.count_nonzero(area & frame.valid)),
        "source_ceiling_percent": round(100 * float(np.count_nonzero(bright)) / req, 3),
        "source_floor_percent": round(100 * float(np.count_nonzero(dark)) / req, 3),
        "unsupported_percent": round(100 * float(np.count_nonzero(missing)) / req, 3),
        "highlight_clusters": fragment_rows(bright, area.shape, 6),
        "shadow_clusters": fragment_rows(dark, area.shape, 6),
        "unsupported_clusters": fragment_rows(missing, area.shape, 6),
    }


def atlas_overlay(frame, area, masks, detail, reliable, ceiling, mode):
    bright, dark, missing = masks
    normal = frame.display() * 0.54
    if mode == "edge_clipping":
        normal[bright] = np.array(STATES["bright"], dtype=np.float32) / 255
        normal[dark] = np.array(STATES["dark"], dtype=np.float32) / 255
    elif mode == "local_detail":
        factor = np.clip(detail[reliable] / max(ceiling, 0.005), 0, 1)[:, None]
        color = np.array(CAPTURE_PALETTE[mode], dtype=np.float32) / 255
        normal[reliable] = (1 - 0.76 * factor) * normal[reliable] + 0.76 * factor * color
    else:
        supported = area & frame.valid
        color = np.array(CAPTURE_PALETTE[mode], dtype=np.float32) / 255
        normal[supported] = normal[supported] * 0.5 + color * 0.5
    normal[missing] = np.array(STATES["absent"], dtype=np.float32) / 255
    alpha = np.where(frame.valid | missing, 1.0, 0.0)
    return uint8_image(normal, alpha)


def add_layer_labels(picture, label, statuses):
    output = picture.convert("RGB")
    width, height = output.size
    margin = max(int(width / 85), 8)
    top = height + 54
    atlas = Image.new("RGB", (width, top), (25, 36, 34))
    atlas.paste(output, (0, 0))
    draw = ImageDraw.Draw(atlas)
    label_font = font(max(12, min(21, int(width / 43))))
    number_font = font(max(11, min(16, int(width / 54))))
    draw.text((margin, height + 5), label, fill=(242, 238, 223), font=label_font)
    small = " · ".join(f"{value:.2f}% {name}" for name, value in statuses)
    draw.text((margin, height + 29), small, fill=(188, 203, 193), font=number_font)
    return atlas


def visible_role(tile, photo_id):
    item = next((photo for photo in tile["photos"] if photo["id"] == photo_id), None)
    if item is None:
        return None
    return item["role"]


def quality_manifest(store, specimen_id, payload):
    request = dictionary(payload, "Photo quality inspection")
    tile = store.tile(specimen_id)
    photo_id = identity(request["photo_id"], "Photo ID") if request.get("photo_id") else None
    selected = active_region(tile, request.get("region_id"))
    grid = integer(request.get("grid", 5), "Quality grid", lower=3, upper=10)
    mode = enumeration(request.get("mode", "edge_clipping"), CAPTURE_PALETTE, "Overlay mode")
    frame = specimen_frame(store.blobs, tile, photo_id)
    area = region_domain(frame.extent, tile, selected)
    basic = focus_diagnostics(frame, area, grid)
    masks = brightness_class(frame, area)
    basic.update(quantity_rows(frame, area, *masks))
    detail, reliable, ceiling = inspected_detail(frame, area)
    basic["native_channel_stats"] = channel_statistics(frame.display(), frame.valid & area)
    basic["captures"] = exposure_records(tile)
    basic["recorded_exposure"] = pair_exposure_span(basic["captures"])
    basic["source_id"] = frame.photo["id"]
    basic["source_role"] = visible_role(tile, frame.photo["id"])
    basic["region_name"] = selected["name"] if selected else "Entire specimen plane"
    picture = atlas_overlay(frame, area, masks, detail, reliable, ceiling, mode)
    labels = [
        ("bright", basic["source_ceiling_percent"]),
        ("dark", basic["source_floor_percent"]),
        ("missing support", basic["unsupported_percent"]),
    ]
    description = mode.replace("_", " ").capitalize()
    return basic, add_layer_labels(picture, description, labels)
