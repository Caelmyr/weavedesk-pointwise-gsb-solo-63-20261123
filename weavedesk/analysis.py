import numpy as np

from .art import (
    annotated_regions,
    base_rgba,
    coverage_overlay,
    magnitude_overlay,
    profile_overlay,
    reviewed_markers,
    signed_overlay,
    stored_preview,
)
from .color import (
    capture_basis,
    compare_reference,
    corrected_linear,
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
from .lighting import measure_direction, measure_glare
from .model import ANALYSIS_LABELS, InputError, library_fields
from .parameters import TOOLS, analysis_parameters
from .profile import measure_profile
from .dots import measure_dots
from .store import key_for
from .texture import measure_texture
from .weave import measure_weave


ANALYZER_VERSION = 2


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


def reference_color_sample(store, tile, photo_id, chosen_region, apply_neutral):
    frame = specimen_frame(store.blobs, tile, photo_id)
    region = active_region(tile, chosen_region)
    domain = region_domain(frame.extent, tile, region)
    valid = analysis_mask(frame, domain, minimum=80, allow_clipped=False)
    linear, correction = corrected_linear(tile, frame, apply_neutral)
    return {
        "fingerprint": reference_fingerprint(linear, valid),
        "basis": capture_basis(tile, frame, correction),
        "region": region,
        "source": capture_provenance(frame),
        "specimen": specimen_provenance(tile),
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
        return {"label": "Fraction of parallel intensity", "low": 0, "high": scale, "midpoint": None, "mode": "sequential"}
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
        fingerprint = reference_fingerprint(linear, support)
        basis = capture_basis(tile, frame, correction)
        metrics = compare_reference(fingerprint, reference["fingerprint"], basis, reference["basis"])
        metrics["reference_title"] = reference["title"]
        metrics["reference_id"] = reference["id"]
        metrics["reference_image"] = reference["annotated_digest"]
        metrics["reference_region"] = reference["region"]
        map_values = reference_map(linear, reference["fingerprint"])
        scale = formula_scale(map_values, support, 12)
        return metrics, magnitude_overlay(frame, map_values, support, scale), theoretical_legend(kind, scale)
    raise RuntimeError(f"Unsupported image measurement: {kind}.")


def compute_pair(tile, first, second, area, params):
    kind = params["kind"]
    options = params["options"]
    if kind == "glare":
        ties = active_region(tile, options["balance_region"])
        calibration_area = area if ties is None else region_domain(first.extent, tile, ties)
        metrics, map_values, valid, unknown = measure_glare(first, second, area, calibration_area, options)
        scale = max(options["glare_fraction"] * 2, 0.45)
        image = magnitude_overlay(first, map_values, valid, scale, unknown=unknown)
        return metrics, image, theoretical_legend(kind, scale)
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
        metrics, picture, legend = compute_pair(tile, frame, pair, area, params)
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
            "reference": {
                "id": reference["id"],
                "version": reference["version"],
                "specimen": reference["specimen"],
                "capture": reference["source"],
                "basis": reference["basis"],
            } if reference else None,
        },
        "pictures": {
            "source": stored_preview(base_rgba(frame, linear), store.blobs),
            "overlay": stored_preview(picture, store.blobs),
            "paired": stored_preview(base_rgba(pair), store.blobs) if pair else None,
        },
    }
    return store.save_result(cache_key, result)
