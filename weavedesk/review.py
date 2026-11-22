import math

import numpy as np
from PIL import ImageDraw

from .art import ACCENT, INK, SHADOW, font, load_artifact_picture, stored_preview
from .model import InputError, collection, dictionary, enumeration, string


REVIEW_COLOR = {"confirm": (162, 225, 171), "review": ACCENT, "dismiss": (159, 162, 154)}
REVIEW_PRIORITY = {"review": 0, "confirm": 1, "dismiss": 2}


def marker_position(bounds, image_size, padding):
    width, height = image_size
    left = int(math.floor(bounds[0] * width)) - padding
    top = int(math.floor(bounds[1] * height)) - padding
    right = int(math.ceil(bounds[2] * width)) + padding
    bottom = int(math.ceil(bounds[3] * height)) + padding
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def block_radius(width, height, bbox):
    centerline = max(6, round(min(width, height) / 85))
    return max(centerline, round(min(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.75))


def marker_meta(record, labels):
    observations = record["metrics"]["observations"]
    calculated = []
    for observation in observations:
        key = observation["key"]
        mark = labels.get(key)
        status = mark["decision"] if mark else "review"
        calculated.append({
            **observation,
            "decision": status,
            "notes": mark.get("notes", "") if mark else "",
        })
    return calculated


def candidate_filter(markers, mode, query):
    result = []
    terms = query.casefold().strip()
    for marker in markers:
        if mode != "all" and marker["decision"] != mode:
            continue
        if terms and not any(terms in value.casefold() for value in (marker["key"], marker["type"], marker["notes"])):
            continue
        result.append(marker)
    result.sort(key=lambda item: (REVIEW_PRIORITY[item["decision"]], -item["strength"], item["key"]))
    return result


def disp_mark(draw, image_size, marker):
    width, height = image_size
    x, y = marker["normalized_center"]
    coords = round(x * width), round(y * height)
    bbox = marker_position(marker["bounds"], image_size, 2)
    radius = block_radius(width, height, bbox)
    position = (coords[0] - radius, coords[1] - radius, coords[0] + radius, coords[1] + radius)
    line = max(1, round(min(width, height) / 325))
    tone = REVIEW_COLOR[marker["decision"]]
    draw.ellipse(position, outline=SHADOW, width=line + 2)
    draw.ellipse(position, outline=tone, width=line)
    if marker["decision"] == "dismiss":
        draw.line((position[0], position[1], position[2], position[3]), fill=tone, width=line)
        draw.line((position[0], position[3], position[2], position[1]), fill=tone, width=line)
    return (*coords, radius)


def draw_label(draw, image_size, center, marker, seen):
    x, y, radius = center
    width, height = image_size
    label = marker["key"]
    typeface = font(max(11, min(19, round(min(width, height) / 41))))
    length = int(math.ceil(draw.textlength(label, font=typeface)))
    placements = [
        (x + radius + 3, y - 7),
        (x - radius - length - 6, y - 7),
        (x - length // 2, y - radius - 16),
        (x - length // 2, y + radius + 2),
    ]
    for px, py in placements:
        if px <= 3 or px + length >= width - 3 or py < 2 or py + 15 >= height - 2:
            continue
        box = (px - 2, py - 1, px + length + 3, py + 13)
        separated = all(box[0] >= item[2] or item[0] >= box[2] or box[1] >= item[3] or item[1] >= box[3] for item in seen)
        if not separated:
            continue
        shade = REVIEW_COLOR[marker["decision"]]
        draw.rounded_rectangle(box, radius=3, fill=SHADOW)
        draw.text((px, py), label, font=typeface, fill=shade)
        seen.append(box)
        return True
    return False


def redraw_review(blobs, record, decisions, *, mode="all", query=""):
    image = load_artifact_picture(blobs, record["pictures"]["source"])
    draw = ImageDraw.Draw(image)
    all_markers = marker_meta(record, decisions)
    markers = candidate_filter(all_markers, mode, query)
    labels = []
    for mark in markers:
        if mark["decision"]=="dismiss" and mode=="all":
            continue
        position = disp_mark(draw, image.size, mark)
        if len(markers) <= 70:
            draw_label(draw, image.size, position, mark, labels)
    return image


def parse_reviews(payload):
    request = dictionary(payload, "Observation review")
    choices = collection(request.get("entries"), "Observation reviews", minimum=1, maximum=160)
    result = {}
    for line in choices:
        item = dictionary(line, "Review decision")
        key = string(item.get("key"), "Observation ID", maximum=8)
        if key in result:
            raise InputError("Each observation can be reviewed once per request.")
        choice = enumeration(item.get("decision"), REVIEW_COLOR, "Decision")
        notes = string(item.get("notes"), "Observation notes", maximum=240, required=False)
        result[key] = {"decision": choice, "notes": notes}
    return result


def observation_balance(markers, shape=None):
    counts = {name: 0 for name in REVIEW_COLOR}
    occupied = np.zeros((3, 3), dtype=np.int32)
    for marker in markers:
        status = marker["decision"]
        counts[status] += 1
        if status == "dismiss":
            continue
        u, v = marker["normalized_center"]
        occupied[min(int(v * 3), 2), min(int(u * 3), 2)] += 1
    return {
        "counts": counts,
        "trace_distribution": occupied.tolist(),
        "extent": list(shape) if shape else None,
    }


def save_reviews(store, measurement_id, expected, changes):
    record = store.result(measurement_id)
    store.require_version(record, expected)
    if record["kind"] != "defects":
        raise InputError("This measurement has no spot observations to review.")
    detector_keys = {entry["key"] for entry in record["metrics"]["observations"]}
    if any(key not in detector_keys for key in changes):
        raise InputError("One or more observations are no longer in this measurement.")
    decisions = {**record["decisions"], **changes}
    artwork = redraw_review(store.blobs, record, decisions)
    summary = observation_balance(marker_meta(record, decisions), artwork.size)
    digest = stored_preview(artwork, store.blobs)
    return store.save_decisions(measurement_id, expected, changes, digest, summary)


def review_selection(record, mode, query):
    if record["kind"] != "defects":
        raise InputError("This measurement has no spot observations.")
    all_markers = marker_meta(record, record["decisions"])
    markers = candidate_filter(all_markers, mode, query)
    return {**observation_balance(all_markers), "items": markers, "filtered_count": len(markers)}
