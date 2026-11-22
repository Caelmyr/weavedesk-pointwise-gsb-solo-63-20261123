"""Follow original spot observations into a later capture without renumbering them."""

import math

from .model import (
    InputError,
    collection,
    dictionary,
    enumeration,
    identity,
    model_version,
    string,
    timestamp,
)


STATUSES = ("confirmed", "tentative", "missing", "unassigned")
STATUS_LABELS = {
    "confirmed": "Confirmed correspondence",
    "tentative": "Tentative correspondence",
    "missing": "Missing evidence",
    "unassigned": "Not linked",
}
MAX_ENTRIES = 160
MAX_TARGETS = 25
EDGE_MARGIN = 0.008
CHANGE_MARGIN = 0.05
PIXEL_MARGIN = 3.0


def parse_correspondence(payload):
    data = dictionary(payload, "Observation correspondence")
    version = model_version(data)
    rows = collection(data.get("entries"), "Correspondence entries", minimum=1, maximum=MAX_ENTRIES)
    entries = {}
    for row in rows:
        item = dictionary(row, "Correspondence entry")
        key = string(item.get("key"), "Original observation ID", maximum=8)
        if key in entries:
            raise InputError("Each original observation can be linked only once per request.")
        status = enumeration(item.get("status"), STATUSES, "Correspondence status")
        note = string(item.get("note", ""), "Correspondence note", maximum=240, required=False)
        target_id = identity(item.get("target_result_id"), "Later measurement") if item.get("target_result_id") else None
        target_key = string(item.get("target_key"), "Later observation ID", maximum=8) if item.get("target_key") else None
        if status == "unassigned":
            target_id = None
            target_key = None
        elif status == "missing":
            if target_key is not None:
                raise InputError("A missing-evidence entry cannot name a later observation.")
        elif target_id is None or target_key is None:
            raise InputError("A matched correspondence needs a later measurement and observation.")
        entries[key] = {
            "target_result_id": target_id,
            "target_key": target_key,
            "status": status,
            "note": note,
        }
    return version, entries


def observation_keys(record):
    return {entry["key"] for entry in record["metrics"]["observations"]}


def observation_entry(record, observation):
    decision = record.get("decisions", {}).get(observation["key"], {})
    return {
        "key": observation["key"],
        "type": observation["type"],
        "polarity": observation["polarity"],
        "area_pixels": observation["area_pixels"],
        "area_mm2": observation.get("area_mm2"),
        "diameter_pixels": observation.get("diameter_pixels"),
        "normalized_center": observation["normalized_center"],
        "bounds": observation["bounds"],
        "decision": decision.get("decision", "review"),
    }


def capture_source(record):
    photo = record["provenance"]["photo"]
    return {
        "id": photo["id"],
        "display_name": photo["display_name"],
        "role": photo["role"],
        "original_digest": photo["original_digest"],
        "upright_size": photo["upright_size"],
        "plane_to_capture": photo["plane_to_capture"],
    }


def source_reading(record, observation):
    return {
        "result_id": record["id"],
        "label": record["label"],
        "created_at": record["created_at"],
        "source_version": record["source_version"],
        "photo": capture_source(record),
        "observation": observation_entry(record, observation),
        "overlay_url": f"/api/results/{record['id']}/image/overlay",
        "source_url": f"/api/results/{record['id']}/image/source",
    }


def measurement_brief(record):
    return {
        "id": record["id"],
        "label": record["label"],
        "created_at": record["created_at"],
        "source_version": record["source_version"],
        "photo": capture_source(record),
        "observation_count": len(record["metrics"]["observations"]),
        "overlay_url": f"/api/results/{record['id']}/image/overlay",
        "source_url": f"/api/results/{record['id']}/image/source",
    }


def validate_target(record, target, target_id):
    if target["id"] == record["id"]:
        raise InputError("Choose a different measurement for the later capture.")
    if target["kind"] != "defects":
        raise InputError("Correspondence targets must be surface observation measurements.")
    if target["tile_id"] != record["tile_id"]:
        raise InputError("Correspondence targets must belong to this specimen.")


def merged_correspondence(store, record, changes):
    source_keys = observation_keys(record)
    unknown = sorted(key for key in changes if key not in source_keys)
    if unknown:
        raise InputError(f"Observation {unknown[0]} is not in this measurement.")
    stored = record.get("correspondence", {})
    merged = {key: dict(value) for key, value in stored.items() if isinstance(value, dict)}
    for key, entry in changes.items():
        if entry["status"] == "unassigned":
            merged.pop(key, None)
        else:
            merged[key] = {**entry, "updated_at": timestamp()}
    cache = {record["id"]: record}

    def target_record(target_id):
        if target_id not in cache:
            cache[target_id] = store.result(target_id)
        return cache[target_id]

    occupied = {}
    for key, entry in merged.items():
        if key not in source_keys:
            raise InputError(f"Observation {key} is no longer in this measurement.")
        if entry["status"] == "missing":
            if entry.get("target_result_id"):
                validate_target(record, target_record(entry["target_result_id"]), entry["target_result_id"])
            continue
        target_id = entry.get("target_result_id")
        target_key = entry.get("target_key")
        if target_id is None or target_key is None:
            raise InputError("A matched correspondence needs a later measurement and observation.")
        target = target_record(target_id)
        validate_target(record, target, target_id)
        if target_key not in observation_keys(target):
            raise InputError(f"Observation {target_key} is not in the later measurement.")
        slot = (target_id, target_key)
        if slot in occupied:
            raise InputError(
                f"Later observation {target_key} is already used by {occupied[slot]}; "
                "one later observation cannot carry two original IDs."
            )
        occupied[slot] = key
    return merged


def save_correspondence(store, measurement_id, expected_version, changes):
    record = store.result(measurement_id)
    if record["kind"] != "defects":
        raise InputError("This measurement has no spot observations to follow.")
    store.require_version(record, expected_version)
    merged = merged_correspondence(store, record, changes)
    if merged == record.get("correspondence", {}):
        return record
    return store.save_correspondence(measurement_id, expected_version, merged)


def edge_contact(observation):
    left, top, right, bottom = observation["bounds"]
    return left <= EDGE_MARGIN or top <= EDGE_MARGIN or right >= 1 - EDGE_MARGIN or bottom >= 1 - EDGE_MARGIN


def comparable_reading(first, second, before, after):
    source = first["provenance"]["specimen"]["physical_mm"]
    later = second["provenance"]["specimen"]["physical_mm"]
    if before.get("area_mm2") is not None and after.get("area_mm2") is not None and source == later:
        return "area_mm2", "mm2"
    extent = first["support"]["registered_image_extent"]
    if extent == second["support"]["registered_image_extent"]:
        return "area_pixels", "px2"
    return None, None


def pixel_area(record, unit):
    if unit == "px2":
        return 1.0
    support = record["support"]
    calibrated = support.get("calibrated_area_mm2")
    pixels = support.get("available_pixels")
    return calibrated / pixels if calibrated and pixels else None


def size_change(first, second, before, after, status):
    if after is None:
        return {"state": None, "reason": "No later observation is linked."}
    if status == "tentative":
        return {
            "state": None,
            "reason": "Tentative correspondence; visible extent is not compared until both worn edges are confirmed.",
        }
    if edge_contact(before):
        return {"state": None, "reason": "The original observation reaches the captured area edge."}
    if edge_contact(after):
        return {"state": None, "reason": "The later observation reaches the captured area edge."}
    field, unit = comparable_reading(first, second, before, after)
    if field is None:
        return {"state": None, "reason": "The two captures do not share a comparable area basis."}
    start = before.get(field)
    stop = after.get(field)
    if start is None or stop is None:
        return {"state": None, "reason": "The worn edge has no measurable area in both captures."}
    per_pixel = pixel_area(second, unit)
    if per_pixel is None:
        return {"state": None, "reason": "The later capture has no supported area calibration."}
    tolerance = max(abs(start) * CHANGE_MARGIN, PIXEL_MARGIN * per_pixel)
    delta = stop - start
    if delta > tolerance:
        state = "enlarged"
    elif delta < -tolerance:
        state = "shrunk"
    else:
        state = "similar"
    return {
        "state": state,
        "unit": unit,
        "before": round(float(start), 5),
        "after": round(float(stop), 5),
        "delta": round(float(delta), 5),
        "delta_percent": round(100 * delta / start, 2) if start else None,
        "tolerance": round(float(tolerance), 5),
        "reason": "Both worn edges are visible in the two captures.",
    }


def center_offset(before, after):
    dx = after["normalized_center"][0] - before["normalized_center"][0]
    dy = after["normalized_center"][1] - before["normalized_center"][1]
    return [round(dx, 6), round(dy, 6)], round(math.hypot(dx, dy), 6)


def target_rows(record, targets, used):
    return [
        {
            **measurement_brief(target),
            "observations": [
                {**observation_entry(target, observation), "used_by": used.get((target["id"], observation["key"]))}
                for observation in target["metrics"]["observations"]
            ],
        }
        for target in targets
    ]


def load_correspondence(store, measurement_id):
    record = store.result(measurement_id)
    if record["kind"] != "defects":
        raise InputError("This measurement has no spot observations to follow.")
    stored = record.get("correspondence", {})
    if not isinstance(stored, dict):
        stored = {}
    available = store.results(record["tile_id"])
    candidates = [item for item in available if item["kind"] == "defects" and item["id"] != record["id"]]
    cache = {item["id"]: item for item in available}
    for entry in stored.values():
        target_id = entry.get("target_result_id")
        if target_id and target_id not in cache:
            cache[target_id] = store.result(target_id)
    candidates = candidates[:MAX_TARGETS]
    known = {item["id"] for item in candidates}
    for entry in stored.values():
        target_id = entry.get("target_result_id")
        if target_id and target_id in cache and target_id not in known:
            candidates.append(cache[target_id])
            known.add(target_id)
    used = {}
    for key, entry in stored.items():
        if entry.get("target_result_id") and entry.get("target_key"):
            used[(entry["target_result_id"], entry["target_key"])] = key
    observations = {item["key"]: item for item in record["metrics"]["observations"]}
    entries = []
    for key in (item["key"] for item in record["metrics"]["observations"]):
        before = observations[key]
        mapping = stored.get(key)
        if mapping is None:
            entries.append({
                "key": key,
                "status": "unassigned",
                "status_label": STATUS_LABELS["unassigned"],
                "note": "",
                "before": source_reading(record, before),
                "after": None,
                "after_measurement": None,
                "change": {"state": None, "reason": "No later observation is linked."},
                "normalized_offset": None,
                "normalized_distance": None,
            })
            continue
        status = mapping.get("status", "missing")
        target = cache.get(mapping.get("target_result_id")) if mapping.get("target_result_id") else None
        target_observation = None
        if target is not None and mapping.get("target_key"):
            target_observation = next(
                (item for item in target["metrics"]["observations"] if item["key"] == mapping["target_key"]),
                None,
            )
        after = source_reading(target, target_observation) if target is not None and target_observation else None
        offset, distance = center_offset(before, target_observation) if target_observation else (None, None)
        if target_observation is None:
            if status == "missing":
                reason = ("No matching worn edge is visible in the checked later capture."
                          if target is not None else "Recorded as missing without a later capture to check.")
            elif target is not None:
                reason = "The linked later observation is no longer in that measurement."
            else:
                reason = "No later observation is linked."
            change = {"state": None, "reason": reason}
        else:
            change = size_change(record, target, before, target_observation, status)
        entries.append({
            "key": key,
            "status": status,
            "status_label": STATUS_LABELS.get(status, status),
            "note": mapping.get("note", ""),
            "before": source_reading(record, before),
            "after": after,
            "after_measurement": measurement_brief(target) if target is not None else None,
            "change": change,
            "normalized_offset": offset,
            "normalized_distance": distance,
        })
    return {
        "measurement": {
            **measurement_brief(record),
            "version": record["version"],
            "photo": capture_source(record),
        },
        "targets": target_rows(record, candidates, used),
        "entries": entries,
        "counts": {
            status: sum(entry["status"] == status for entry in entries)
            for status in STATUSES
        },
    }
