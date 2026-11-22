import numpy as np

from .art import (
    annotated_regions,
    base_rgba,
    coverage_overlay,
    magnitude_overlay,
    polarized_overlay,
    profile_overlay,
    reviewed_markers,
    signed_overlay,
    stored_preview,
)
from .color import (
    capture_basis,
    compare_reference,
    corrected_linear,
    delta_e,
    linear_lab,
    quantile_rows,
    reference_fingerprint,
    reference_map,
    summarize_colors,
)
from .condition import aggregate_candidates, scan_marking, survey_joint
from .coverage import measure_coverage
from .blade import blade_width
from .defects import measure_defects
from .craze import measure_craze
from .rim import measure_rim
from .geometry import active_region, analysis_mask, region_domain, specimen_frame
from .glare_rule import (
    current_line,
    example_fields,
    fit_adopted_line,
    line_reference,
    measure_polarized,
    pair_light,
    same_line,
    session_states,
)
from .lighting import measure_direction
from .masks import restored_mask, restored_plane, stored_mask, stored_plane
from .model import ANALYSIS_LABELS, InputError, dictionary, identity, library_fields, model_version
from .parameters import TOOLS, analysis_parameters
from .profile import measure_profile
from .dots import measure_dots
from .store import key_for
from .texture import measure_texture
from .weave import measure_weave


ANALYZER_VERSION = 3
MIN_REFERENCE_PIXELS = 40


def selected_capture(tile, explicit, expected_role=None):
    captures = tile["photos"]
    plane = tile.get("plane")
    if not plane:
        raise InputError("Add a specimen photograph before measuring.")
    registered = {plane["photo_id"], *tile["registrations"]}
    if explicit:
        found = next((item for item in captures if item["id"] == explicit), None)
        if found is None:
            raise InputError("The selected photo is no longer attached to this specimen.")
        if found["id"] not in registered:
            raise InputError("Register this photo in the tile plane before measuring it.")
        if expected_role and found["role"] != expected_role:
            raise InputError(f"Choose a {expected_role} photo for this measurement.")
        return found["id"]
    if expected_role is None:
        return plane["photo_id"]
    choices = [item["id"] for item in captures if item["role"] == expected_role and item["id"] in registered]
    if not choices:
        raise InputError(f"Add and register a {expected_role} photo before running this measurement.")
    if len(choices) > 1:
        raise InputError(f"Choose which registered {expected_role} photograph to measure.")
    return choices[0]


def capture_provenance(frame):
    source = frame.photo
    return {
        "id": source["id"],
        "display_name": source["name"],
        "role": source["role"],
        "original_digest": source["source_digest"],
        "working_digest": source["working_digest"],
        "raw_size": source["original_size"],
        "upright_size": [source["width"], source["height"]],
        "working_color_space": source["working_color_space"],
        "plane_to_capture": list(frame.transform),
        "synthetic": source["synthetic"],
    }


def specimen_provenance(tile):
    return {
        "id": tile["id"],
        "version": tile["version"],
        "name": tile["name"],
        "body": tile["body"],
        "glaze": tile["glaze"],
        "firing": tile["firing"],
        "batch": tile["batch"],
        "physical_mm": [tile["width_mm"], tile["height_mm"]],
    }


def support_digest(frame, area):
    total = int(np.count_nonzero(area))
    available = area & frame.valid
    taken = int(np.count_nonzero(available))
    clipped = int(np.count_nonzero(available & frame.clipped))
    return {
        "registered_image_extent": list(frame.extent),
        "requested_pixels": total,
        "available_pixels": taken,
        "clipped_pixels": clipped,
        "available_percent": round(100 * taken / total, 2) if total else 0,
        "calibrated_area_mm2": round(taken * frame.scale[0] * frame.scale[1], 3) if frame.scale else None,
    }


def adopt_glare_line(store, tile_id, payload):
    data = dictionary(payload, "Glare line")
    version = model_version(data)
    parallel_id = identity(data.get("parallel_id"), "Parallel-polarized photo")
    cross_id = identity(data.get("cross_id"), "Cross-polarized photo")
    if parallel_id == cross_id:
        raise InputError("Choose two different polarized photos.")
    current = store.tile(tile_id)
    store.require_version(current, version)
    examples = example_fields(data)
    parallel = specimen_frame(store.blobs, current, parallel_id)
    cross = specimen_frame(store.blobs, current, cross_id)
    if parallel.photo["role"] != "parallel" or cross.photo["role"] != "cross":
        raise InputError("Adopt the line from one parallel-polarized and one cross-polarized capture.")
    record = fit_adopted_line(current, parallel, cross, examples)

    def change(tile):
        tile["glare_line"] = record

    return store.edit_tile(tile_id, version, change, "glare", "Adopted the specimen's single polarized glare line")


def clear_glare_line(store, tile_id, payload):
    version = model_version(dictionary(payload, "Glare line removal"))

    def change(tile):
        tile["glare_line"] = None

    return store.edit_tile(tile_id, version, change, "glare", "Cleared the specimen's polarized glare line")


def line_pair_frames(store, tile, line):
    photos = {item["id"]: item for item in tile["photos"]}
    pair = line["pair"]
    ids = (pair["primary_id"], pair["comparison_id"])
    if any(identifier not in photos for identifier in ids):
        raise InputError("A photo used by the adopted glare line has been removed. Adopt the line again.")
    parallel = specimen_frame(store.blobs, tile, ids[0])
    cross = specimen_frame(store.blobs, tile, ids[1])
    if parallel.photo["role"] != "parallel" or cross.photo["role"] != "cross":
        raise InputError("The lighting labels of the adopted glare line changed. Adopt the line again.")
    return parallel, cross


def region_crop(frame, domain):
    rows, cols = np.nonzero(domain)
    if not len(rows):
        raise InputError("The selected region has no supported pixels.")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    return (left, top, right - left, bottom - top)


def reference_color_sample(store, tile, photo_id, chosen_region, apply_neutral):
    region = active_region(tile, chosen_region)
    line = tile.get("glare_line")
    if line is None:
        return plain_reference_sample(store, tile, photo_id, chosen_region, region, apply_neutral)
    parallel, cross = line_pair_frames(store, tile, line)
    plane = region_domain(parallel.extent, tile, None)
    light, _ = pair_light(parallel, cross, plane)
    states = session_states(light, line["line"])
    domain = region_domain(parallel.extent, tile, region)
    valid = analysis_mask(parallel, domain, minimum=80, allow_clipped=False)
    used = valid & states.retained
    if np.count_nonzero(used) < MIN_REFERENCE_PIXELS:
        raise InputError("Too little of this reference patch is reliable under the specimen's glare line.")
    linear, correction = corrected_linear(tile, parallel, apply_neutral)
    left, top, width, height = region_crop(parallel, domain)
    crop = np.s_[top:top + height, left:left + width]
    raw_lab = linear_lab(linear)
    return {
        "fingerprint": reference_fingerprint(linear, used),
        "basis": capture_basis(tile, parallel, correction),
        "region": region,
        "source": capture_provenance(parallel),
        "specimen": specimen_provenance(tile),
        "glare_line": line_reference(line),
        "session_visible": stored_mask(store.blobs, valid),
        "session_retained": stored_mask(store.blobs, used),
        "session_plane": stored_plane(store.blobs, raw_lab[crop]),
        "session_bounds": [left, top, width, height],
        "glare_excluded_percent": round(
            100 * float(np.count_nonzero(states.unreliable & valid)) / max(int(np.count_nonzero(valid)), 1), 2
        ),
        "annotated_digest": stored_preview(
            annotated_regions(parallel, [region] if region else [], linear), store.blobs
        ),
    }


def plain_reference_sample(store, tile, photo_id, chosen_region, region, apply_neutral):
    frame = specimen_frame(store.blobs, tile, photo_id)
    domain = region_domain(frame.extent, tile, region)
    valid = analysis_mask(frame, domain, minimum=80, allow_clipped=False)
    linear, correction = corrected_linear(tile, frame, apply_neutral)
    return {
        "fingerprint": reference_fingerprint(linear, valid),
        "basis": capture_basis(tile, frame, correction),
        "region": region,
        "source": capture_provenance(frame),
        "specimen": specimen_provenance(tile),
        "glare_line": None,
        "session_visible": None,
        "session_retained": None,
        "session_plane": None,
        "session_bounds": None,
        "glare_excluded_percent": None,
        "annotated_digest": stored_preview(annotated_regions(frame, [region] if region else [], linear), store.blobs),
    }


def approve_reference(store, payload):
    fields = library_fields(payload)
    tile = store.tile(fields["tile_id"])
    reference = reference_color_sample(store, tile, None, fields["region_id"], True)
    return store.add_reference({**fields, **reference})


def resolve_parameters(store, tile, payload):
    params = analysis_parameters(payload)
    kind = params["kind"]
    setting = TOOLS[kind]
    roles = setting.get("capture_pair")
    if kind == "reference" and tile.get("glare_line"):
        roles = ("parallel", "cross")
    params["photo_id"] = selected_capture(tile, params["photo_id"], roles[0] if roles else None)
    params["comparison_id"] = selected_capture(tile, params["comparison_id"], roles[1]) if roles else None
    if not roles and payload.get("comparison_id"):
        raise InputError("This measurement uses one capture.")
    reference = None
    if kind == "reference":
        reference = store.reference(params["options"]["reference_id"])
        if reference["retired"]:
            raise InputError("The selected reference has been retired.")
    if kind == "defects" and params["options"]["maximum_area"] <= params["options"]["minimum_area"]:
        raise InputError("Maximum spot size must exceed the minimum.")
    return params, reference


def theoretical_legend(kind, scale=None):
    if kind == "color" or kind == "reference":
        return {"label": "ΔE76", "low": 0, "high": scale, "midpoint": None, "mode": "sequential"}
    if kind == "texture":
        return {"label": "Local intensity RMS", "low": 0, "high": scale, "midpoint": None, "mode": "sequential"}
    if kind == "glare":
        return {"mode": "categorical", "labels": [
            {"name": "Color retained", "color": "#5ba88f"},
            {"name": "Light unreliable", "color": "#f08a5d"},
            {"name": "Too little detail", "color": "#8f958c"},
            {"name": "Not visible", "color": "#353a3a"},
        ]}
    if kind == "direction":
        return {"label": "Raking contrast · right to left", "low": -scale, "high": scale, "midpoint": 0, "mode": "signed"}
    if kind == "coverage":
        return {"mode": "categorical", "labels": [
            {"name": "Bare", "color": "#ebb25e"},
            {"name": "Mixed", "color": "#dad892"},
            {"name": "Textile", "color": "#5ba88f"},
            {"name": "Unmeasured", "color": "#353a3a"},
        ]}
    return None


def formula_scale(values, support, floor):
    available = values[support]
    return max(float(np.quantile(available, 0.97)), floor) if len(available) else floor


def compute_single(store, tile, frame, region, area, linear, correction, params, reference):
    kind = params["kind"]
    options = params["options"]
    support = analysis_mask(frame, area, allow_clipped=False)
    if kind == "dots":
        metrics, image = measure_dots(frame, area, linear, options)
        return metrics, image, None
    if kind == "weave":
        metrics, image = measure_weave(frame, linear, area, options)
        return metrics, image, None
    if kind == "blade":
        metrics, image = blade_width(tile, frame, area, linear, options)
        return metrics, image, None
    if kind in {"marking", "joint", "aggregate"}:
        operation = {"marking": scan_marking, "joint": survey_joint, "aggregate": aggregate_candidates}[kind]
        metrics, image = operation(frame, area, linear, options)
        return metrics, image, None
    if kind == "color":
        metrics, map_values = summarize_colors(linear, support, options["grid"])
        scale = formula_scale(map_values, support, 6)
        image = magnitude_overlay(frame, map_values, support, scale, source=linear, unknown=area & ~support)
        return metrics, image, theoretical_legend(kind, scale)
    if kind == "coverage":
        unrestricted = region_domain(frame.extent, tile)
        metrics, states = measure_coverage(tile, frame, linear, area, unrestricted, options)
        return metrics, coverage_overlay(frame, states, area), theoretical_legend(kind)
    if kind == "defects":
        metrics = measure_defects(frame, linear, area, options)
        image = reviewed_markers(frame, metrics["observations"], source=linear)
        return metrics, image, None
    if kind == "texture":
        metrics, roughness, valid = measure_texture(frame, linear, area, options)
        scale = formula_scale(roughness, valid, 0.01)
        return metrics, magnitude_overlay(frame, roughness, valid, scale), theoretical_legend(kind, scale)
    if kind == "craze":
        metrics, image = measure_craze(frame, linear, area, options)
        return metrics, image, None
    if kind == "rim":
        metrics, image = measure_rim(frame, linear, area, options)
        return metrics, image, None
    if kind == "profile":
        metrics = measure_profile(frame, linear, area, options)
        image = profile_overlay(frame, metrics["endpoints"], options["band_pixels"], linear)
        return metrics, image, None
    if kind == "reference":
        if tile.get("glare_line"):
            raise RuntimeError("A specimen glare line requires the reference pair path.")
        return plain_reference_match(tile, frame, region, support, linear, correction, params, reference)
    raise RuntimeError(f"Unsupported image measurement: {kind}.")


def plain_reference_match(tile, frame, region, support, linear, correction, params, reference):
    if reference.get("glare_line"):
        raise InputError("This reference was approved under the specimen's glare line, which is now cleared or missing. Restore the line or approve the reference again.")
    fingerprint = reference_fingerprint(linear, support)
    metrics = compare_reference(
        fingerprint,
        reference["fingerprint"],
        capture_basis(tile, frame, correction),
        reference["basis"],
    )
    metrics.update({
        "reference_title": reference["title"],
        "reference_id": reference["id"],
        "reference_image": reference["annotated_digest"],
        "reference_region": reference["region"],
        "glare_line": None,
    })
    map_values = reference_map(linear, reference["fingerprint"])
    scale = formula_scale(map_values, support, 12)
    return metrics, magnitude_overlay(frame, map_values, support, scale), theoretical_legend("reference", scale)


def reference_match_pair(tile, parallel, cross, region, params, reference, store):
    line = current_line(tile)
    stored = reference.get("glare_line")
    if not stored:
        raise InputError("This reference was approved without the specimen's glare line. Approve it again before re-checking.")
    if not same_line(stored["line"], line["line"]):
        raise InputError("This reference was approved under a different glare line. Approve it again under the current line.")
    if not reference.get("session_retained") or not reference.get("session_visible") or not reference.get("session_plane"):
        raise InputError("This reference has no retained old-session record. Approve it again under the glare line.")
    plane = region_domain(parallel.extent, tile, None)
    light, _ = pair_light(parallel, cross, plane)
    states = session_states(light, line["line"])
    domain = region_domain(parallel.extent, tile, region)
    valid = analysis_mask(parallel, domain, minimum=80, allow_clipped=False)
    old_visible = restored_mask(store.blobs, reference["session_visible"], parallel.extent)
    old_retained = restored_mask(store.blobs, reference["session_retained"], parallel.extent)
    left, top, width, height = reference["session_bounds"]
    crop = np.s_[top:top + height, left:left + width]
    approved = restored_plane(store.blobs, reference["session_plane"], reference["session_bounds"])
    linear, correction = corrected_linear(tile, parallel, params["apply_neutral"])
    current = linear_lab(linear)
    crop_delta = delta_e(current[crop], approved)
    distances = np.zeros(valid.shape, dtype=np.float32)
    distances[crop] = crop_delta
    inside = np.zeros(valid.shape, dtype=bool)
    inside[crop] = True
    judged = valid & states.retained & old_retained & inside
    judged_count = int(np.count_nonzero(judged))
    if judged_count < MIN_REFERENCE_PIXELS:
        raise InputError("Too few positions are visible and reliable in both the approved sample and this photo pair.")
    floor = params["options"]["delta_e_floor"]
    changed = judged & (distances >= floor)
    excluded_light = valid & states.unreliable & old_visible
    excluded_blank = valid & ~old_visible
    excluded_detail = valid & old_visible & states.too_little_detail
    excluded_other = valid & ~judged & ~excluded_light & ~excluded_blank & ~excluded_detail
    metrics = compare_reference(
        reference_fingerprint(linear, judged),
        reference["fingerprint"],
        capture_basis(tile, parallel, correction),
        reference["basis"],
    )
    metrics.update({
        "reference_title": reference["title"],
        "reference_id": reference["id"],
        "reference_image": reference["annotated_digest"],
        "reference_region": reference["region"],
        "glare_line": line_reference(line),
        "judged_pixels": judged_count,
        "judged_percent": round(100 * judged_count / max(int(np.count_nonzero(valid)), 1), 2),
        "position_delta_e_quantiles": quantile_rows(distances[judged]),
        "position_delta_e_median": round(float(np.median(distances[judged])), 3),
        "light_unreliable_percent": round(100 * float(np.count_nonzero(excluded_light)) / max(int(np.count_nonzero(valid)), 1), 2),
        "old_left_blank_percent": round(100 * float(np.count_nonzero(excluded_blank)) / max(int(np.count_nonzero(valid)), 1), 2),
        "too_little_detail_percent": round(100 * float(np.count_nonzero(excluded_detail)) / max(int(np.count_nonzero(valid)), 1), 2),
        "outside_old_region_percent": round(100 * float(np.count_nonzero(excluded_other)) / max(int(np.count_nonzero(valid)), 1), 2),
        "noticeable_change_percent": round(100 * float(np.count_nonzero(changed)) / judged_count, 2),
        "noticeable_change_delta_e": floor,
        "note": "Color difference is read only where both sessions are visible and admitted by the one adopted glare line.",
    })
    scale = formula_scale(np.where(inside, distances, 0.0), judged, 12)
    picture = magnitude_overlay(
        parallel,
        np.where(judged, distances, 0.0).astype(np.float32),
        judged,
        scale,
        source=linear,
        unknown=valid & ~judged,
    )
    return metrics, picture, theoretical_legend("reference", scale)


def compute_pair(store, tile, first, second, area, params, reference):
    kind = params["kind"]
    options = params["options"]
    if kind == "glare":
        line = current_line(tile)
        metrics, states, domain = measure_polarized(first, second, area, line, options)
        return metrics, polarized_overlay(first, states, domain), theoretical_legend(kind)
    if kind == "reference":
        return reference_match_pair(tile, first, second, params["region"], params, reference, store)
    metrics, values, valid, unknown = measure_direction(first, second, area, options)
    scale = max(metrics["local_threshold"] * 2.4, 0.06)
    image = signed_overlay(first, values, valid, scale, unknown=unknown)
    return metrics, image, theoretical_legend(kind, scale)


def result_key(tile, params, reference):
    return key_for({
        "analyzer_version": ANALYZER_VERSION,
        "params": params,
        "tile_id": tile["id"],
        "tile_version": tile["version"],
        "reference_version": reference["version"] if reference else None,
    })


def evaluate(store, tile_id, payload):
    tile = store.tile(tile_id)
    params, reference = resolve_parameters(store, tile, payload)
    cache_key = result_key(tile, params, reference)
    cached = store.get_cached_result(cache_key)
    if cached:
        return cached
    frame = specimen_frame(store.blobs, tile, params["photo_id"])
    region = active_region(tile, params["region_id"])
    area = region_domain(frame.extent, tile, region)
    pair = specimen_frame(store.blobs, tile, params["comparison_id"]) if params["comparison_id"] else None
    correction = None
    linear = frame.linear
    if pair is not None:
        params["region"] = region
        metrics, picture, legend = compute_pair(store, tile, frame, pair, area, params, reference)
    else:
        linear, correction = corrected_linear(tile, frame, params["apply_neutral"])
        metrics, picture, legend = compute_single(store, tile, frame, region, area, linear, correction, params, reference)
    result = {
        "tile_id": tile_id,
        "source_version": tile["version"],
        "kind": params["kind"],
        "label": ANALYSIS_LABELS[params["kind"]],
        "parameters": params,
        "metrics": metrics,
        "legend": legend,
        "basis": capture_basis(tile, frame, correction),
        "support": support_digest(frame, area),
        "provenance": {
            "analyzer_version": ANALYZER_VERSION,
            "specimen": specimen_provenance(tile),
            "photo": capture_provenance(frame),
            "paired_photo": capture_provenance(pair) if pair else None,
            "region": region,
            "exclusions": [item for item in tile["regions"] if item["purpose"] == "ignore"],
            "neutral_card": correction,
            "glare_line": line_reference(tile.get("glare_line")),
            "reference": {
                "id": reference["id"],
                "version": reference["version"],
                "specimen": reference["specimen"],
                "capture": reference["source"],
                "basis": reference["basis"],
                "glare_line": reference.get("glare_line"),
            } if reference else None,
        },
        "pictures": {
            "source": stored_preview(base_rgba(frame, linear), store.blobs),
            "overlay": stored_preview(picture, store.blobs),
            "paired": stored_preview(base_rgba(pair), store.blobs) if pair else None,
        },
    }
    return store.save_result(cache_key, result)
