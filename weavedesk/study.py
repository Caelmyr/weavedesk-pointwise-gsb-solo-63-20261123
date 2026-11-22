from collections import defaultdict
import math

from .art import comparison_sheet, load_artifact_picture, stored_preview
from .model import InputError, dictionary, enumeration, string, unique_ids


STUDY_CHOICES = {
    "color": {
        "median_lightness": ("Survey plane median L*", lambda result: result["metrics"]["median_lab"][0]),
        "variation_p90": ("Color variation p90 ΔE76", lambda result: result["metrics"]["delta_e_from_tile_median"]["90"]),
    },
    "coverage": {
        "glazed": ("Blade % of measured area", lambda result: result["metrics"]["percent"]["glazed"]),
        "mixed": ("Mixed % of measured area", lambda result: result["metrics"]["percent"]["mixed"]),
    },
    "glare": {
        "glare": ("Strong glare %", lambda result: result["metrics"]["strong_glare_percent"]),
    },
    "texture": {
        "median_detail": ("Median roughness in image intensity", lambda result: result["metrics"]["grid_roughness"]["50"]),
    },
}


def comparable_signature(result):
    basis = result["basis"]
    metrics = result["metrics"]
    options = result["parameters"]["options"]
    mode = result["kind"]
    schema = {
        "photo_role": basis["photo_role"],
        "synthetic": basis["synthetic"],
        "neutral_target": basis["neutral_target"],
        "source_space": basis["source_space"],
    }
    if mode == "coverage":
        schema["ambiguity"] = options["ambiguity"]
        schema["off_axis_limit"] = options["max_residual_fraction"]
    if mode == "glare":
        schema["balance"] = options["balance"]
        schema["glare_fraction"] = options["glare_fraction"]
        schema["comparison_role"] = metrics["pair"]["comparison_role"]
    if mode == "texture":
        schema["radius"] = options["background_radius"]
        schema["edge_scale"] = options["edge_smooth"]
    return schema


def group_label(result, group_by):
    specimen = result["provenance"]["specimen"]
    chosen = specimen[group_by]
    return chosen.strip() or "Unlabeled"


def capture_note(differences):
    if not differences:
        return "Selected measurements use the same lighting labels and method settings."
    return "Groups mix " + ", ".join(differences) + ". Use comparisons as descriptive image observations."


def trial_entry(record, selector, group_by):
    source = record["provenance"]["specimen"]
    value = selector(record)
    if value is None or not math.isfinite(value):
        raise InputError("One selected run does not have a numeric value for this comparison.")
    return {
        "measurement_id": record["id"],
        "tile_id": source["id"],
        "tile_name": source["name"],
        "group": group_label(record, group_by),
        "version": source["version"],
        "value": round(float(value), 6),
        "capture": record["provenance"]["photo"],
        "overlay": record["pictures"]["overlay"],
        "support": record["support"]["available_percent"],
    }


def study_stats(entries):
    statistics = []
    grouped = defaultdict(list)
    for item in entries:
        grouped[item["group"]].append(item)
    for label, group in sorted(grouped.items(), key=lambda pair: pair[0].casefold()):
        points = [item["value"] for item in group]
        average = sum(points) / len(points)
        sum_sq = sum((point - average) ** 2 for point in points)
        stats = {
            "group": label,
            "n": len(points),
            "mean": round(average, 6),
            "minimum": round(min(points), 6),
            "maximum": round(max(points), 6),
            "spread": round(math.sqrt(sum_sq / (len(points) - 1)), 6) if len(points) > 1 else None,
        }
        statistics.append(stats)
    return statistics


def assess_sampling_groups(entries, group_by):
    grouped = defaultdict(list)
    group_frame = {}
    for sample in entries:
        series = sample["group"]
        grouped[series].append(sample["support"])
    for label, points in grouped.items():
        group_frame[label] = {
            "lower_support_percent": min(points),
            "upper_support_percent": max(points),
        }
    only_one = [label for label, points in grouped.items() if len(points) == 1]
    caveats = []
    if only_one:
        caveats.append("One measurement in " + ", ".join(only_one) + ".")
    if any(float(item["support"]) < 92 for item in entries):
        caveats.append("Image support differs from the requested region for some specimens.")
    if len(grouped) == 1:
        caveats.append(f"Only one '{group_by}' group appears in this selection.")
    return group_frame, caveats


def sheets_for_study(blobs, title, entries, unit):
    digests = []
    for offset in range(0, len(entries), 9):
        group = entries[offset:offset + 9]
        cards = []
        for item in group:
            source = load_artifact_picture(blobs, item["overlay"])
            cards.append({
                "image": source,
                "heading": item["tile_name"],
                "description": f"{item['group']} · {item['value']:.3f} · {unit}",
            })
        label = title if len(entries) <= 9 else f"{title} · {offset + 1}–{offset + len(cards)}"
        digests.append(stored_preview(comparison_sheet(cards, label), blobs))
    return digests


def create_study(store, payload):
    data = dictionary(payload, "Firing comparison")
    title = string(data.get("title"), "Study title", maximum=100)
    group_by = enumeration(data.get("group_by", "firing"), ("firing", "body", "glaze", "batch"), "Group by")
    identifiers = unique_ids(data.get("result_ids"), "Measurements", minimum=2, maximum=18)
    records = [store.result(result_id) for result_id in identifiers]
    kind = records[0]["kind"]
    if kind not in STUDY_CHOICES or any(record["kind"] != kind for record in records):
        raise InputError("Select 2 to 18 runs of the same supported measurement type.")
    if len({record["tile_id"] for record in records}) != len(records):
        raise InputError("Select one measurement per specimen for a firing comparison.")
    option = enumeration(data.get("metric"), STUDY_CHOICES[kind], "Metric")
    unit, reducer = STUDY_CHOICES[kind][option]
    signatures = [comparable_signature(record) for record in records]
    differences = sorted(key.replace("_", " ") for key in signatures[0] if len({item[key] for item in signatures}) > 1)
    entries = sorted(
        [trial_entry(record, reducer, group_by) for record in records],
        key=lambda item: (item["group"].casefold(), item["tile_name"].casefold(), item["measurement_id"]),
    )
    sampling, caveats = assess_sampling_groups(entries, group_by)
    return store.save_study({
        "title": title,
        "kind": kind,
        "metric": option,
        "metric_label": unit,
        "group_by": group_by,
        "rows": entries,
        "groups": study_stats(entries),
        "sampling": sampling,
        "comparison_note": capture_note(differences),
        "caveats": caveats,
        "pictures": sheets_for_study(store.blobs, title, entries, unit),
    })


def public_metrics():
    return {kind: {key: descriptor[0] for key, descriptor in options.items()} for kind, options in STUDY_CHOICES.items()}
