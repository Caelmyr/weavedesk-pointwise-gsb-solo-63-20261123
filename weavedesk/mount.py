import math

import numpy as np
from PIL import Image, ImageDraw

from .analysis import capture_provenance, specimen_provenance
from .art import ACCENT, INK, SHADOW, base_rgba, font, stored_preview
from .geometry import polygon_mask, specimen_frame
from .model import (
    REGION_PURPOSES,
    InputError,
    collection,
    dictionary,
    identity,
    integer,
    scalar,
    string,
)


CLEAR_VOTE = 0.75
ZONE_CLEAR_FRACTION = 0.8
MAX_UNCOVERED_SPANS = 12
SECURE_OUTLINE = (244, 172, 99)
DISPLAY_OUTLINE = (150, 220, 190)
LOSS_TONE = (210, 96, 108)
EXCLUDED_TONE = (159, 162, 154)
BAND_TINT = (38, 58, 66)
PENDING_TONE = (232, 164, 84)
CONFIRM_TONE = (162, 225, 171)


def pick_region(tile, region_id, purpose, label):
    found = next((item for item in tile["regions"] if item["id"] == region_id), None)
    if found is None:
        raise InputError(f"The {label} region is no longer on this specimen.")
    if found["purpose"] != purpose:
        raise InputError(f"Mark the {label} region with the purpose “{REGION_PURPOSES[purpose]}”.")
    return found


def plan_fields(payload, height_mm):
    data = dictionary(payload, "Clamp frame plan")
    bar_width = scalar(data.get("bar_width_mm"), "Bar contact width", lower=0.5, upper=min(200.0, height_mm))
    stops = [
        scalar(item, "Detent position", lower=0, upper=height_mm)
        for item in collection(data.get("stops_mm"), "Detent positions", minimum=2, maximum=6)
    ]
    requested = data.get("stop")
    return {
        "title": string(data.get("title"), "Plan title", maximum=90, required=False) or "Clamp frame plan",
        "bar_width_mm": bar_width,
        "bar_gap_mm": scalar(data.get("bar_gap_mm"), "Bar spacing", lower=bar_width, upper=2 * height_mm),
        "stops_mm": stops,
        "min_contact_fraction": scalar(
            data.get("min_contact_fraction", 0.5), "Minimum fabric contact", lower=0.05, upper=1,
        ),
        "stop": integer(requested, "Detent", lower=1, upper=len(stops)) if requested is not None else None,
    }


def row_runs(rows):
    if not len(rows):
        return []
    runs = []
    start = previous = rows[0]
    for row in rows[1:]:
        if row != previous + 1:
            runs.append((start, previous))
            start = row
        previous = row
    runs.append((start, previous))
    return runs


def band_rows(height, mm_per_px, y_mm):
    first = max(0, int(np.ceil(y_mm[0] / mm_per_px - 0.5)))
    last = min(height - 1, int(np.floor(y_mm[1] / mm_per_px - 0.5)))
    return None if last < first else (first, last)


def uncovered_spans(mask, covered, mm_per_px, pixel_mm2):
    exposed = mask & ~covered
    spans = []
    for first, last in row_runs(np.nonzero(exposed.any(axis=1))[0])[:MAX_UNCOVERED_SPANS]:
        spans.append({
            "y_mm": [round(first * mm_per_px, 2), round((last + 1) * mm_per_px, 2)],
            "area_mm2": round(int(np.count_nonzero(exposed[first:last + 1])) * pixel_mm2, 2),
        })
    return spans


def evaluate_stop(index, center_mm, fields, height_mm, extent, masks):
    width_mm, gap = fields["bar_width_mm"], fields["bar_gap_mm"]
    fabric, secure_fabric, display_fabric, loss_mask, mm_per_px, pixel_mm2 = masks
    width, height = extent
    bars = []
    reasons = []
    union = np.zeros((height, width), dtype=bool)
    for bar_index, center in enumerate((center_mm, center_mm + gap), 1):
        y_mm = (center - width_mm / 2, center + width_mm / 2)
        info = {
            "index": bar_index,
            "y_mm": [round(y_mm[0], 3), round(y_mm[1], 3)],
            "y_norm": [round(y_mm[0] / height_mm, 6), round(y_mm[1] / height_mm, 6)],
        }
        rows = None
        if y_mm[0] < -1e-9 or y_mm[1] > height_mm + 1e-9:
            reasons.append(f"Bar {bar_index} ({y_mm[0]:.1f}–{y_mm[1]:.1f} mm) falls outside the fabric plane.")
        else:
            rows = band_rows(height, mm_per_px, y_mm)
            if rows is None:
                reasons.append(f"Bar {bar_index} ({y_mm[0]:.1f}–{y_mm[1]:.1f} mm) covers no image rows.")
        contact_fraction = 0.0
        if rows is not None:
            band = np.zeros((height, width), dtype=bool)
            band[rows[0]:rows[1] + 1, :] = True
            union |= band
            contact = band & fabric
            contact_fraction = float(np.mean(contact[band]))
            if contact_fraction < fields["min_contact_fraction"]:
                reasons.append(
                    f"Bar {bar_index} presses surviving fabric on only {contact_fraction * 100:.0f}% of its band "
                    f"(needs {fields['min_contact_fraction'] * 100:.0f}%)."
                )
            info.update(
                contact_mm2=round(int(np.count_nonzero(contact)) * pixel_mm2, 2),
                pressed_secure_mm2=round(int(np.count_nonzero(band & secure_fabric)) * pixel_mm2, 2),
                label_covered_mm2=round(int(np.count_nonzero(band & display_fabric)) * pixel_mm2, 2),
                loss_mm2=round(int(np.count_nonzero(band & loss_mask)) * pixel_mm2, 2),
            )
        info["contact_fraction"] = round(contact_fraction, 4)
        info.setdefault("contact_mm2", 0.0)
        info.setdefault("pressed_secure_mm2", 0.0)
        info.setdefault("label_covered_mm2", 0.0)
        info.setdefault("loss_mm2", 0.0)
        bars.append(info)
    secure_total = int(np.count_nonzero(secure_fabric))
    label_total = int(np.count_nonzero(display_fabric))
    return {
        "index": index,
        "installable": not reasons,
        "reasons": reasons,
        "bars": bars,
        "secure_coverage_percent": round(100 * int(np.count_nonzero(union & secure_fabric)) / secure_total, 2),
        "label_exposed_percent": round(100 * (1 - int(np.count_nonzero(union & display_fabric)) / label_total), 2),
        "uncovered_secure": uncovered_spans(secure_fabric, union, mm_per_px, pixel_mm2),
    }


def region_points(region, extent):
    width, height = extent
    return [(x * width, y * height) for x, y in region["vertices"]]


def outline_region(draw, region, extent, color, stroke):
    points = region_points(region, extent)
    draw.line([*points, points[0]], fill=SHADOW, width=stroke + 2, joint="curve")
    draw.line([*points, points[0]], fill=color, width=stroke, joint="curve")


def plan_overlay(frame, secure, display, losses, excluded, stops, adopted_index):
    image = base_rgba(frame)
    width, height = image.size
    tint = Image.new("RGBA", image.size, (0, 0, 0, 0))
    painter = ImageDraw.Draw(tint)
    adopted = stops[adopted_index - 1]
    for bar in adopted["bars"]:
        painter.rectangle((0, bar["y_norm"][0] * height, width, bar["y_norm"][1] * height), fill=BAND_TINT + (88,))
    for region in losses:
        painter.polygon(region_points(region, image.size), fill=LOSS_TONE + (104,))
    image = Image.alpha_composite(image, tint)
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(min(width, height) / 320))
    for region in excluded:
        outline_region(draw, region, image.size, EXCLUDED_TONE, stroke)
    for region in losses:
        outline_region(draw, region, image.size, LOSS_TONE, stroke)
    for region, color in ((secure, SECURE_OUTLINE), (display, DISPLAY_OUTLINE)):
        outline_region(draw, region, image.size, color, stroke)
        x, y = region_points(region, image.size)[0]
        draw.text((x + 4, y + 3), region["name"][:36], fill=color, stroke_width=2, stroke_fill=SHADOW)
    typeface = font(max(11, min(19, round(min(width, height) / 41))))
    for stop in stops:
        band = stop["bars"][0]["y_norm"]
        y = min(max((band[0] + band[1]) / 2 * height, 8), height - 8)
        chosen = stop["index"] == adopted_index
        color = ACCENT if chosen else EXCLUDED_TONE
        draw.line((width - 14, y, width, y), fill=color, width=stroke + 1)
        label = f"S{stop['index']}"
        draw.text((width - 20 - draw.textlength(label, font=typeface), y - 7), label, font=typeface, fill=color, stroke_width=2, stroke_fill=SHADOW)
    for bar in adopted["bars"]:
        top, bottom = bar["y_norm"][0] * height, bar["y_norm"][1] * height
        draw.line((0, top, width, top), fill=ACCENT, width=stroke + 1)
        draw.line((0, bottom, width, bottom), fill=ACCENT, width=stroke + 1)
        label = f"Bar {bar['index']} · {bar['y_mm'][0]:.1f}–{bar['y_mm'][1]:.1f} mm"
        draw.text((width * 0.03, (top + bottom) / 2 - 8), label, font=typeface, fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image


def plan_mount(store, specimen_id, payload):
    tile = store.tile(specimen_id)
    if tile.get("plane") is None:
        raise InputError("Set a base photograph and mark the specimen plane first.")
    if tile["width_mm"] is None or tile["height_mm"] is None:
        raise InputError("Enter the specimen's physical dimensions before planning a clamp frame.")
    data = dictionary(payload, "Clamp frame plan")
    fields = plan_fields(data, tile["height_mm"])
    secure = pick_region(tile, identity(data.get("secure_region_id"), "Thread-loss risk region"), "fragile", "thread-loss risk")
    display = pick_region(tile, identity(data.get("display_region_id"), "Display label region"), "display", "display label")
    frame = specimen_frame(store.blobs, tile)
    width, height = frame.extent
    mm_per_px = tile["height_mm"] / height
    pixel_mm2 = frame.scale[0] * frame.scale[1]
    losses = [item for item in tile["regions"] if item["purpose"] == "loss"]
    excluded = [item for item in tile["regions"] if item["purpose"] == "ignore"]
    fabric = frame.valid.copy()
    for region in [*losses, *excluded]:
        fabric &= ~polygon_mask(frame.extent, region["vertices"])
    loss_mask = np.zeros((height, width), dtype=bool)
    for region in losses:
        loss_mask |= polygon_mask(frame.extent, region["vertices"])
    secure_fabric = polygon_mask(frame.extent, secure["vertices"]) & fabric
    if not np.count_nonzero(secure_fabric):
        raise InputError("The thread-loss risk area holds no surviving fabric in this plane.")
    display_fabric = polygon_mask(frame.extent, display["vertices"]) & fabric
    if not np.count_nonzero(display_fabric):
        raise InputError("The display label area holds no surviving fabric in this plane.")
    masks = (fabric, secure_fabric, display_fabric, loss_mask, mm_per_px, pixel_mm2)
    stops = [
        evaluate_stop(index, center, fields, tile["height_mm"], frame.extent, masks)
        for index, center in enumerate(fields["stops_mm"], 1)
    ]
    installable = [stop for stop in stops if stop["installable"]]
    if not installable:
        detail = " ".join(f"Stop {stop['index']}: {'; '.join(stop['reasons'])}" for stop in stops)
        raise InputError(f"No detent can be installed with this frame and fabric. {detail}")
    recommended = max(
        installable,
        key=lambda stop: (stop["secure_coverage_percent"], stop["label_exposed_percent"], -stop["index"]),
    )
    adopted = stops[fields["stop"] - 1] if fields["stop"] else recommended
    if not adopted["installable"]:
        raise InputError(f"Stop {adopted['index']} cannot be installed: {'; '.join(adopted['reasons'])}")
    overlay = plan_overlay(frame, secure, display, losses, excluded, stops, adopted["index"])
    return store.save_mount({
        "kind": "plan",
        "tile_id": tile["id"],
        "title": fields["title"],
        "specimen": specimen_provenance(tile),
        "source": capture_provenance(frame),
        "plane": {"size": tile["plane"]["size"], "corners": tile["plane"]["corners"]},
        "parameters": {
            "bar_width_mm": fields["bar_width_mm"],
            "bar_gap_mm": fields["bar_gap_mm"],
            "stops_mm": fields["stops_mm"],
            "min_contact_fraction": fields["min_contact_fraction"],
        },
        "regions": {"secure": secure, "display": display, "losses": losses, "excluded": excluded},
        "stops": stops,
        "recommended_stop": recommended["index"],
        "adopted_stop": adopted["index"],
        "adopted": adopted,
        "overlay": stored_preview(overlay, store.blobs),
        "event_message": f"Planned clamp frame · adopted detent {adopted['index']} of {len(stops)}",
    })


def session_info(record):
    photo = record["provenance"]["photo"]
    return {
        "result_id": record["id"],
        "photo_id": photo["id"],
        "photo_name": photo["display_name"],
        "created_at": record["created_at"],
        "source_version": record["source_version"],
    }


def review_overlay(frame, bands, zones, confirmable, pre_existing, pending):
    image = base_rgba(frame)
    width, height = image.size
    tint = Image.new("RGBA", image.size, (0, 0, 0, 0))
    painter = ImageDraw.Draw(tint)
    for band in bands:
        painter.rectangle((0, band[0] * height, width, band[1] * height), fill=BAND_TINT + (70,))
    for zone in zones:
        painter.rectangle((0, zone["y_norm"][0] * height, width, zone["y_norm"][1] * height), fill=PENDING_TONE + (110,))
    image = Image.alpha_composite(image, tint)
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(min(width, height) / 320))
    radius = max(5, round(min(width, height) / 92))
    typeface = font(max(11, min(19, round(min(width, height) / 41))))
    for band in bands:
        for edge in band:
            draw.line((0, edge * height, width, edge * height), fill=ACCENT, width=stroke)

    def circle(observation, color, label=None):
        x = observation["normalized_center"][0] * width
        y = observation["normalized_center"][1] * height
        box = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(box, outline=SHADOW, width=stroke + 2)
        draw.ellipse(box, outline=color, width=stroke)
        if label:
            draw.text((x + radius + 4, y - 7), label, font=typeface, fill=color, stroke_width=2, stroke_fill=SHADOW)

    for item in pre_existing:
        circle(item, EXCLUDED_TONE)
    for item in pending:
        circle(item, PENDING_TONE)
    label_keys = len(confirmable) <= 35
    for item in confirmable:
        circle(item, CONFIRM_TONE, item["key"] if label_keys else None)
    return image


def review_mount(store, specimen_id, payload):
    tile = store.tile(specimen_id)
    data = dictionary(payload, "Removal review")
    title = string(data.get("title"), "Review title", maximum=90, required=False) or "Clamp frame removal review"
    plan = store.mount(identity(data.get("plan_id"), "Clamp frame plan"))
    if plan["kind"] != "plan" or plan["tile_id"] != tile["id"]:
        raise InputError("Choose a clamp frame plan from this specimen.")
    plane = tile.get("plane")
    if plane is None or plane["size"] != plan["plane"]["size"] or plane["corners"] != plan["plane"]["corners"]:
        raise InputError("The specimen plane changed since this plan was adopted. Re-plan the clamp frame before reviewing.")
    if tile["width_mm"] is None or tile["height_mm"] is None:
        raise InputError("Enter the specimen's physical dimensions before reviewing a clamp frame plan.")
    if [tile["width_mm"], tile["height_mm"]] != plan["specimen"]["physical_mm"]:
        raise InputError("The specimen dimensions changed since this plan was adopted. Re-plan the clamp frame before reviewing.")
    baseline = store.result(identity(data.get("baseline_result_id"), "Earlier spot observations"))
    repeat = store.result(identity(data.get("repeat_result_id"), "Post-removal spot observations"))
    for record, label in ((baseline, "earlier"), (repeat, "post-removal")):
        if record["tile_id"] != tile["id"] or record["kind"] != "defects":
            raise InputError(f"Choose the {label} spot observations from this specimen.")
        if record["support"]["registered_image_extent"] != plane["size"]:
            raise InputError(f"The {label} spot observations were measured against another specimen plane. Re-run them.")
    if baseline["provenance"]["photo"]["id"] == repeat["provenance"]["photo"]["id"]:
        raise InputError("The post-removal observations must come from the re-photographed capture, not the earlier capture.")
    radius = scalar(data.get("match_radius_px", 8), "Match radius", lower=1, upper=60)
    base_frame = specimen_frame(store.blobs, tile, baseline["provenance"]["photo"]["id"])
    repeat_frame = specimen_frame(store.blobs, tile, repeat["provenance"]["photo"]["id"])
    photo_order = {photo["id"]: index for index, photo in enumerate(tile["photos"])}
    if photo_order[baseline["provenance"]["photo"]["id"]] >= photo_order[repeat["provenance"]["photo"]["id"]]:
        raise InputError("The earlier observations must come from a capture added before the post-removal capture. Swap the two selections.")
    width, height = repeat_frame.extent
    mm_per_px = tile["height_mm"] / height
    clear_base = base_frame.valid & ~base_frame.clipped
    clear_repeat = repeat_frame.valid & ~repeat_frame.clipped
    vote_radius = max(2, round(min(width, height) / 200))

    def clear_at(mask, point):
        cx, cy = point[0] * width - 0.5, point[1] * height - 0.5
        window = mask[
            max(0, int(cy - vote_radius)):int(cy + vote_radius) + 1,
            max(0, int(cx - vote_radius)):int(cx + vote_radius) + 1,
        ]
        return bool(window.size) and float(window.mean()) >= CLEAR_VOTE

    def unclear_sessions(point):
        sessions = []
        if not clear_at(clear_base, point):
            sessions.append("baseline")
        if not clear_at(clear_repeat, point):
            sessions.append("repeat")
        return sessions

    adopted_bars = plan["adopted"]["bars"]

    def bar_for(point):
        y = point[1]
        for bar in adopted_bars:
            if bar["y_norm"][0] - 1e-9 <= y <= bar["y_norm"][1] + 1e-9:
                return bar["index"]
        return None

    base_observations = baseline["metrics"]["observations"]
    repeat_observations = repeat["metrics"]["observations"]

    def nearest(point, candidates):
        best, best_distance = None, None
        for observation in candidates:
            dx = (observation["normalized_center"][0] - point[0]) * width
            dy = (observation["normalized_center"][1] - point[1]) * height
            distance = math.hypot(dx, dy)
            if best_distance is None or distance < best_distance:
                best, best_distance = observation, distance
        return best if best_distance is not None and best_distance <= radius else None

    confirmable, pre_existing, outside, pending = [], [], [], []
    for observation in repeat_observations:
        point = observation["normalized_center"]
        bar = bar_for(point)
        if bar is None:
            outside.append({"key": observation["key"], "normalized_center": point, "type": observation["type"]})
            continue
        match = nearest(point, base_observations)
        if match is not None:
            pre_existing.append({
                "repeat_key": observation["key"],
                "baseline_key": match["key"],
                "normalized_center": point,
                "bar": bar,
                "area_mm2": observation["area_mm2"],
                "clear_in_both": not unclear_sessions(point),
            })
            continue
        unclear = unclear_sessions(point)
        if unclear:
            pending.append({
                "entry": "observation", "key": observation["key"], "normalized_center": point,
                "bar": bar, "unclear_in": unclear,
            })
        else:
            confirmable.append({
                "key": observation["key"], "type": observation["type"], "polarity": observation["polarity"],
                "area_mm2": observation["area_mm2"], "normalized_center": point, "bar": bar,
                "strength": observation["strength"],
            })
    baseline_under = []
    for observation in base_observations:
        point = observation["normalized_center"]
        bar = bar_for(point)
        if bar is None:
            continue
        baseline_under.append({
            "key": observation["key"], "normalized_center": point, "bar": bar,
            "seen_in_repeat": nearest(point, repeat_observations) is not None,
        })
        if not clear_at(clear_repeat, point):
            pending.append({
                "entry": "baseline_observation", "key": observation["key"], "normalized_center": point,
                "bar": bar, "unclear_in": ["repeat"],
            })
    fabric = base_frame.valid.copy()
    for region in [*plan["regions"]["losses"], *plan["regions"]["excluded"]]:
        fabric &= ~polygon_mask(repeat_frame.extent, region["vertices"])
    zones = []
    for bar in adopted_bars:
        rows = band_rows(height, mm_per_px, bar["y_mm"])
        if rows is None:
            continue
        obscured = []
        for row in range(rows[0], rows[1] + 1):
            cloth = fabric[row]
            if cloth.any() and float(np.mean((clear_base[row] & clear_repeat[row])[cloth])) < ZONE_CLEAR_FRACTION:
                obscured.append(row)
        for first, last in row_runs(obscured):
            block = fabric[first:last + 1]
            unclear_in = [
                name for name, mask in (("baseline", clear_base), ("repeat", clear_repeat))
                if float(np.mean(mask[first:last + 1][block])) < ZONE_CLEAR_FRACTION
            ] or ["baseline", "repeat"]
            zones.append({
                "entry": "zone",
                "bar": bar["index"],
                "y_mm": [round(first * mm_per_px, 2), round((last + 1) * mm_per_px, 2)],
                "y_norm": [round(first / height, 6), round((last + 1) / height, 6)],
                "unclear_in": unclear_in,
            })
    pending.extend(zones)

    def pending_order(item):
        y = item["normalized_center"][1] if "normalized_center" in item else item["y_norm"][0]
        return item["bar"], y

    pending.sort(key=pending_order)
    for group in (confirmable, pre_existing, baseline_under, outside):
        group.sort(key=lambda item: (item.get("bar", 0), item["normalized_center"][1], item["normalized_center"][0]))
    counts = {
        "confirmable": len(confirmable),
        "pre_existing": len(pre_existing),
        "outside_bars": len(outside),
        "pending_recapture": len(pending),
        "baseline_under_bars": len(baseline_under),
    }
    overlay = review_overlay(
        repeat_frame,
        [bar["y_norm"] for bar in adopted_bars],
        zones,
        confirmable,
        pre_existing,
        [item for item in pending if "normalized_center" in item],
    )
    return store.save_mount({
        "kind": "review",
        "tile_id": tile["id"],
        "title": title,
        "specimen": specimen_provenance(tile),
        "plane": {"size": plane["size"], "corners": plane["corners"]},
        "plan_id": plan["id"],
        "plan_adopted": {
            "plan_title": plan["title"],
            "plan_created_at": plan["created_at"],
            "adopted_stop": plan["adopted_stop"],
            "parameters": plan["parameters"],
            "bars": [{"index": bar["index"], "y_mm": bar["y_mm"], "y_norm": bar["y_norm"]} for bar in adopted_bars],
        },
        "baseline": session_info(baseline),
        "repeat": session_info(repeat),
        "match_radius_px": radius,
        "counts": counts,
        "confirmable": confirmable,
        "pre_existing": pre_existing,
        "outside_bars": outside,
        "baseline_under_bars": baseline_under,
        "pending_recapture": pending,
        "overlay": stored_preview(overlay, store.blobs),
        "event_message": f"Reviewed clamp frame removal · {len(confirmable)} confirmable change(s)",
    })


def summarize_mount(record):
    summary = {key: record[key] for key in ("id", "kind", "title", "created_at")}
    if record["kind"] == "plan":
        summary.update(
            adopted_stop=record["adopted_stop"],
            stop_count=len(record["stops"]),
            secure_coverage_percent=record["adopted"]["secure_coverage_percent"],
            label_exposed_percent=record["adopted"]["label_exposed_percent"],
        )
    else:
        summary.update(plan_id=record["plan_id"], counts=record["counts"])
    return summary
