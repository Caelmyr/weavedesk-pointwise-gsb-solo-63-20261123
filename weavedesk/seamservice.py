"""Ribbon seam service: request validation, rectification and persistence."""

import numpy as np

from .art import stored_preview
from .model import InputError, dictionary, identity, scalar, string
from .photos import open_working
from .seam import (
    SIDES,
    annotation_overlay,
    composite_band,
    evaluate_notches,
    line_scan,
    notch_step_from_payload,
    parse_fragment_payload,
    parse_scan_line,
    rectify_fragment,
)
from .seamdemo import demo_photo_records, demo_specimen, render_demo_pair


def _photo_lookup(store, photo_id):
    wanted = identity(photo_id, "Fragment photo ID")
    for tile in store.list_tiles():
        for photo in tile["photos"]:
            if photo["id"] == wanted:
                return tile, photo
    raise InputError("A fragment photograph is not attached to any specimen. Upload it under Captures first.")


def _band_dimensions(photo, corners, pixels_per_mm):
    points = np.asarray(corners, dtype=np.float64) * [photo["width"], photo["height"]]
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    band_width_px = float(np.mean(lengths[::2]))
    band_height_px = float(np.mean(lengths[1::2]))
    if min(band_width_px, band_height_px) < 40:
        raise InputError("Marked ribbon band is too small to rectify.")
    width_mm = band_width_px / pixels_per_mm
    height_mm = band_height_px / pixels_per_mm
    # Keep both fragments on one shared physical pixel grid.
    common_ppm = max(1.0, min(pixels_per_mm, 8.0))
    return (
        max(40, int(round(width_mm * common_ppm))),
        max(40, int(round(height_mm * common_ppm))),
        common_ppm,
    )


def _load_fragment(store, side, payload):
    parsed = parse_fragment_payload(side, payload)
    tile, photo = _photo_lookup(store, parsed["photo_id"])
    pixels_per_mm = scalar(payload.get("pixels_per_mm", 4.0), f"{side} pixels per mm", lower=0.2, upper=60.0)
    band_width, band_height, common_ppm = _band_dimensions(photo, parsed["corners"], pixels_per_mm)
    rgb, alpha = open_working(store.blobs, photo)
    support = alpha >= 0.995
    fragment = rectify_fragment(
        side, photo, rgb, support, parsed["corners"], parsed["outline"],
        band_width, band_height, common_ppm,
    )
    return fragment, {
        "photo_id": photo["id"],
        "photo_name": photo["name"],
        "tile_id": tile["id"],
        "corners": parsed["corners"],
        "outline": parsed["outline"],
        "pixels_per_mm": pixels_per_mm,
        "band_size": [band_width, band_height],
        "synthetic": bool(photo.get("synthetic")),
    }


def create_seam(store, payload):
    data = dictionary(payload, "Seam request")
    title = string(data.get("title"), "Seam title", maximum=90)
    fragments = {}
    meta = {}
    for side in SIDES:
        if side not in data:
            raise InputError(f"Include the {side} fragment specification.")
        fragments[side], meta[side] = _load_fragment(store, side, data[side])
    left, right = fragments["left"], fragments["right"]
    if abs(left.height - right.height) > max(left.height, right.height) * 0.18:
        raise InputError("The two rectified bands differ too much in width. Check the ribbon corners.")
    notch_step_px = notch_step_from_payload(data, default_px=30.0)
    chosen, report, evidence = evaluate_notches(left, right, notch_step_px)
    chosen_shift = chosen["shift_px"]
    canvas, ownership, supported, layout = composite_band(left, right, chosen_shift)
    overlay = annotation_overlay(canvas, ownership, layout, left, right, chosen, evidence, chosen_shift)
    record = {
        "title": title,
        "chosen_notch": chosen["notch"],
        "chosen_label": chosen["label"],
        "notch_step_px": notch_step_px,
        "shift_px": chosen_shift,
        "notch_report": report,
        "evidence": _public_evidence(evidence),
        "fragments": meta,
        "layout": {key: value for key, value in layout.items() if key != "seam_x"},
        "provenance": {
            "left_photo": _photo_provenance(left),
            "right_photo": _photo_provenance(right),
            "rule": "Pixels are copied from one fragment only; the gap and unphotographed tear are never filled.",
        },
        "line_scans": [],
    }
    saved = store.save_seam(record)
    saved["composite_digest"] = stored_preview(overlay, store.blobs)
    saved = _update_seam(store, saved)
    return saved


def _update_seam(store, record):
    from .store import document

    with store.transaction(write=True) as conn:
        conn.execute("UPDATE seams SET document=? WHERE id=?", (document(record), record["id"]))
    return record


def _public_evidence(evidence):
    def slim(items):
        return [{key: (round(value, 4) if isinstance(value, float) else value)
                 for key, value in item.items() if key not in ("x_mean",)} for item in items]
    return {
        "left_stripes": slim(evidence["left_stripes"]),
        "right_stripes": slim(evidence["right_stripes"]),
        "left_threads": slim(evidence["left_threads"]),
        "right_threads": slim(evidence["right_threads"]),
        "thread_tolerance_px": evidence["thread_tolerance_px"],
    }


def _photo_provenance(fragment):
    return {
        "id": fragment.photo["id"],
        "name": fragment.photo["name"],
        "source_digest": fragment.photo["source_digest"],
        "synthetic": bool(fragment.photo.get("synthetic")),
        "pixels_per_mm": fragment.pixels_per_mm,
    }


def preview_notch(store, seam_id, payload):
    record = store.seam(seam_id)
    data = dictionary(payload, "Notch preview")
    notch = int(data.get("notch", record["chosen_notch"]))
    if notch not in (-1, 0, 1):
        raise InputError("Notch must be -1, 0 or 1.")
    left, right = _rebuild_fragments(store, record)
    shift = notch * record["notch_step_px"]
    canvas, ownership, supported, layout = composite_band(left, right, shift)
    chosen = next(item for item in record["notch_report"] if item["notch"] == notch)
    overlay = annotation_overlay(canvas, ownership, layout, left, right, chosen, _rehydrate_evidence(record), shift)
    return overlay, chosen


def _rehydrate_evidence(record):
    evidence = record["evidence"]
    return {
        "left_stripes": evidence["left_stripes"],
        "right_stripes": evidence["right_stripes"],
        "left_threads": evidence["left_threads"],
        "right_threads": evidence["right_threads"],
        "thread_tolerance_px": evidence["thread_tolerance_px"],
    }


def _rebuild_fragments(store, record):
    rebuilt = {}
    for side in SIDES:
        spec = record["fragments"][side]
        tile, photo = _photo_lookup(store, spec["photo_id"])
        rgb, alpha = open_working(store.blobs, photo)
        support = alpha >= 0.995
        band_width, band_height = spec["band_size"]
        rebuilt[side] = rectify_fragment(
            side, photo, rgb, support, spec["corners"], spec["outline"],
            band_width, band_height, spec["pixels_per_mm"],
        )
    return rebuilt["left"], rebuilt["right"]


def add_line_scan(store, seam_id, payload):
    record = store.seam(seam_id)
    start, end, stations, band_pixels = parse_scan_line(payload)
    left, right = _rebuild_fragments(store, record)
    shift = record["chosen_notch"] * record["notch_step_px"]
    canvas, ownership, supported, layout = composite_band(left, right, shift)
    scan = line_scan(left, right, layout, ownership, supported, start, end, stations, band_pixels)
    record["line_scans"].append(scan)
    return _update_seam(store, record), scan


def load_demo_seam(store):
    pair = render_demo_pair()
    photos = demo_photo_records(store, pair)
    specimen = demo_specimen(store, photos)
    payload = {
        "title": "Broken indigo ribbon · practice seam",
        "notch_step_px": pair["notch_step_px"],
        "left": {
            "photo_id": photos["left"]["photo"]["id"],
            "corners": photos["left"]["corners"],
            "outline": photos["left"]["outline"],
            "pixels_per_mm": photos["left"]["pixels_per_mm"],
        },
        "right": {
            "photo_id": photos["right"]["photo"]["id"],
            "corners": photos["right"]["corners"],
            "outline": photos["right"]["outline"],
            "pixels_per_mm": photos["right"]["pixels_per_mm"],
        },
    }
    return create_seam(store, payload), pair
