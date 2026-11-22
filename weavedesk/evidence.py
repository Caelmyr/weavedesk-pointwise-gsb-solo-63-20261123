import base64
import csv
from html import escape
from io import BytesIO, StringIO
import json
from pathlib import PurePath
import re
from zipfile import ZIP_DEFLATED, ZipFile

from .art import comparison_sheet, load_artifact_picture, stored_preview
from .model import InputError, dictionary, enumeration, flag, string, tags, unique_ids
from .store import document


SOURCE_TOTAL_BYTES = 60 * 1024 * 1024
EXPORT_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def filename_stem(value, fallback):
    clean = EXPORT_NAME_PATTERN.sub("-", value).strip("-_.")
    return clean[:65] or fallback


def csv_lines(headers, rows):
    stream = StringIO(newline="")
    out = csv.writer(stream, lineterminator="\r\n")
    out.writerow(headers)
    for row in rows:
        out.writerow([csv_cell(value) for value in row])
    return stream.getvalue().encode("utf-8-sig")


def csv_cell(value):
    if isinstance(value, str) and value and value[0] in "=+-@\t\r":
        return "'" + value
    return document(value) if isinstance(value, (dict, list)) else value


def result_metrics_table(record):
    kind = record["kind"]
    metrics = record["metrics"]
    if kind == "profile":
        columns = (
            "position", "position_unit", "relative_position", "samples", "lightness", "a", "b", "intensity",
        )
        rows = ([item[key] for key in columns] for item in metrics["rows"])
    elif kind == "defects":
        columns = (
            "key", "type", "polarity", "area_pixels", "area_mm2", "normalized_center", "bounds", "peak_contrast",
            "decision", "notes",
        )
        rows = []
        decisions = record.get("decisions", {})
        for marker in metrics["observations"]:
            decision = decisions.get(marker["key"], {})
            row = [marker[key] for key in columns[:-2]]
            rows.append([*row, decision.get("decision", ""), decision.get("notes", "")])
    elif kind == "coverage":
        columns = ("relative_top", "relative_bottom", "measured", "bare_percent", "mixed_percent", "glazed_percent",
                   "unmeasured_percent")
        rows = (
            [band["relative_top"], band["relative_bottom"], band["measured"]]
            + [band["percent"][key] for key in ("bare", "mixed", "glazed", "unmeasured")]
            for band in metrics["depth_bands"]
        )
    elif kind == "color":
        columns = ("cell", "bounds", "pixels", "lab", "delta_from_center", "within_p90", "swatch")
        rows = ([item[key] for key in columns] for item in metrics["cells"])
    elif kind == "texture":
        columns = ("direction", "spacing_px", "spacing_mm", "sample_pairs", "correlation")
        rows = (
            [curve["direction"], point["spacing_px"], point.get("spacing_mm"), point["sample_pairs"], point["correlation"]]
            for curve in metrics["lag_curves"]
            for point in curve["points"]
        )
    elif kind == "rim":
        columns = ("side", "depth_from", "depth_to", "pixels", "median_lab", "median_from_center_delta_e")
        rows = (
            [side["side"], *entry["fraction_from_edge"], entry["pixels"], entry["median_lab"], entry["median_from_center_delta_e"]]
            for side in metrics["sides"]
            for entry in side["swaths"]
        )
    elif kind == "craze":
        columns = ("cell", "pixels", "area_mm2", "bounds")
        rows = ([i, item["pixels"], item["mm2"], item["bounds"]] for i, item in enumerate(metrics["largest_cells"], 1))
    elif kind == "direction":
        columns = ("cell", "median_signed", "strong_percent", "positive_percent")
        rows = ([item[key] for key in columns] for item in metrics["cells"])
    elif kind == "glare":
        columns = ("pixels", "area_mm2", "bounds")
        rows = ([item[key] for key in columns] for item in metrics["hotspots"])
    elif kind == "marking":
        fields = ("condition", "first_station", "last_station", "stations", "midpoint_normalized", "median_cross_width")
        columns = ("unit", *fields)
        rows = ([metrics["unit"], *(item[key] for key in fields)] for item in metrics["runs"])
    elif kind == "joint":
        fields = ("position", "left", "right", "right_minus_left", "status", "normalized_position")
        columns = ("unit", *fields)
        rows = ([metrics["unit"], *(item[key] for key in fields)] for item in metrics["stations"])
    elif kind == "dots":
        columns = ("dot", "area_px", "area_mm2", "axis_ratio", "major_angle_deg", "center")
        rows = ([item[key] for key in columns] for item in metrics["rows"])
    elif kind == "weave":
        fields = ("family", "angle_degrees", "pitch", "peak_to_background")
        columns = ("unit", *fields)
        rows = ([metrics["unit"], *(item[key] for key in fields)] for item in metrics["rows"])
    elif kind == "blade":
        fields = ("position", "center", "state", "width", "endpoints")
        columns = ("unit", *fields)
        rows = ([metrics["length_unit"], *(item[key] for key in fields)] for item in metrics["stations"])
    elif kind == "aggregate":
        fields = ("rank", "registered_pixels", "normalized_center", "contrast", "bounding_width", "bounding_height")
        columns = ("unit", *fields)
        rows = ([metrics["unit"], *(item[key] for key in fields)] for item in metrics["candidates"])
    else:
        columns = ("metric", "value")
        rows = [[key, val] for key, val in metrics.items() if not isinstance(val, (dict, list))]
    return csv_lines(columns, rows)


def study_table(report):
    columns = ("tile_name", "tile_id", "group", "measurement_id", "version", "value", "support")
    return csv_lines(columns, ([item[key] for key in columns] for item in report["rows"]))


def source_index(tile, results):
    originals = {}
    for result in results:
        provenance = result["provenance"]
        for capture in (provenance["photo"], provenance["paired_photo"]):
            if not capture:
                continue
            digest = capture["original_digest"]
            current = next((item for item in tile["photos"] if item["id"] == capture["id"]), None)
            originals[digest] = {
                "digest": digest,
                "name": capture["display_name"],
                "role": capture["role"],
                "mime": current["original_mime"] if current else None,
            }
    return list(originals.values())


def pinned_fields(store, request):
    data = dictionary(request, "Evidence packet")
    targets = unique_ids(data.get("result_ids"), "Measurement", minimum=1, maximum=12)
    title = string(data.get("title"), "Evidence title", maximum=115)
    notes = string(data.get("notes"), "Evidence notes", maximum=2400, required=False)
    view = enumeration(data.get("view", "overlay"), ("overlay", "source"), "Export image")
    include_sources = flag(data.get("include_sources", False), "Include original uploads")
    return data, targets, title, notes, view, include_sources


def image_card(store, result, view):
    provenance = result["provenance"]
    specimen = provenance["specimen"]
    chosen_picture = load_artifact_picture(store.blobs, result["pictures"][view])
    coverage = float(result["support"]["available_percent"])
    return {
        "image": chosen_picture,
        "heading": f"{result['label']} · {specimen['name']}",
        "description": f"Specimen r{result['source_version']} · {coverage:.1f}% support",
    }


def evidence_sheets(store, title, records, view):
    digests = []
    for offset in range(0, len(records), 6):
        selected = records[offset:offset + 6]
        cards = [image_card(store, result, view) for result in selected]
        label = title if len(records) <= 6 else f"{title} · Sheet {offset // 6 + 1}"
        sheet = comparison_sheet(cards, label)
        digests.append(stored_preview(sheet, store.blobs))
    return digests


def create_evidence(store, identifier, payload):
    data, targets, title, notes, view, include_sources = pinned_fields(store, payload)
    tile, records = store.sources_for_evidence(identifier, targets)
    if any(result["tile_id"] != tile["id"] for result in records):
        raise InputError("An evidence packet can contain measurements from its own specimen only.")
    originals = source_index(tile, records) if include_sources else []
    byte_total = sum(store.blobs.require(item["digest"]).stat().st_size for item in originals)
    if byte_total > SOURCE_TOTAL_BYTES:
        raise InputError("Selected originals exceed the packet limit of 60 MiB. Omit originals or pin fewer measurements.")
    return store.save_evidence({
        "title": title,
        "notes": notes,
        "tags": tags(data.get("tags", [])),
        "tile_id": tile["id"],
        "tile_name": tile["name"],
        "tile_version_at_pin": tile["version"],
        "synthetic": tile["synthetic"],
        "view": view,
        "originals": originals,
        "original_bytes": byte_total,
        "measurements": records,
        "sheets": evidence_sheets(store, title, records, view),
    })


def inline_picture(blobstore, digest):
    return "data:image/png;base64," + base64.b64encode(blobstore.get(digest)).decode("ascii")


def measurement_article(record, sheet):
    metrics = record["metrics"]
    rows = []
    for key, value in metrics.items():
        if not isinstance(value, (str, bool, float, int)) or len(str(value)) > 180:
            continue
        label = key.replace("_", " ")
        rows.append(f"<tr><th>{escape(label)}</th><td>{escape(str(value))}</td></tr>")
    capture = record["provenance"]["photo"]
    title = escape(record["label"])
    location = escape(capture["display_name"])
    rev = record["source_version"]
    summaries = "".join(rows)
    figures = ""
    if sheet:
        figures = f'<img alt="{title} from {location}" src="{sheet}">'
    return (
        f'<article><h2>{title}</h2><p>{location} · specimen revision {rev}</p>'
        f'{figures}<table>{summaries}</table></article>'
    )


def exported_image_name(record, view):
    return f"measurements/{result_entry_names(record)}-{view}.png"


def specimen_html(store, entry, interactive=True):
    title = escape(entry["title"])
    name = escape(entry["tile_name"])
    date = escape(entry["created_at"])
    notes = escape(entry["notes"]).replace("\n", "<br>") or "No inspection notes."
    badge = "<p>Specimen uses synthetic training material.</p>" if entry["synthetic"] else ""
    measurements = []
    for record in entry["measurements"]:
        digest = record["pictures"][entry["view"]]
        url = inline_picture(store.blobs, digest) if interactive else exported_image_name(record, entry["view"])
        measurements.append(measurement_article(record, url))
    body = "".join(measurements)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:860px;margin:50px auto;padding:0 20px;background:#f9f5eb;color:#252e2d}}
h1,h2{{line-height:1.12}}
h1{{font-size:2.5rem}}
article{{border-top:1px solid #c9c5b8;padding:22px 0;break-inside:avoid}}
img{{max-width:100%;max-height:520px;background:#242b2b}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}}
td,th{{border-bottom:1px solid #ccc;text-align:left;padding:0.45rem 0;vertical-align:top}}
</style></head><body><h1>{title}</h1><p>{name} · {date}</p>{badge}<p>{notes}</p>{body}
<footer>WeaveDesk · Capture-relative visual evidence</footer></body></html>""".encode("utf8")


def result_entry_names(record):
    stem = filename_stem(record["label"], "measurement")
    return f"{stem}-{record['id'][:8]}"


def artifact_json(payload):
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True) + "\n").encode("utf8")


def evidence_zip(store, entry):
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=4) as zipfile:
        zipfile.writestr("inspection.html", specimen_html(store, entry, interactive=False))
        zipfile.writestr("manifest.json", artifact_json(entry))
        for index, digest in enumerate(entry["sheets"], 1):
            zipfile.writestr(f"sheets/sheet-{index:02d}.png", store.blobs.get(digest))
        for record in entry["measurements"]:
            name = result_entry_names(record)
            zipfile.writestr(f"measurements/{name}.json", artifact_json(record))
            zipfile.writestr(f"measurements/{name}.csv", result_metrics_table(record))
            for view, digest in record["pictures"].items():
                if digest:
                    zipfile.writestr(exported_image_name(record, view), store.blobs.get(digest))
        for source in entry["originals"]:
            suffix = PurePath(source["name"]).suffix.lower().strip(".")
            suffix = suffix if suffix in ("png", "jpg", "jpeg", "tif", "tiff", "webp") else "img"
            basename = filename_stem(PurePath(source["name"]).stem, "original")
            file = store.blobs.require(source["digest"])
            zipfile.write(file, f"originals/{basename}-{source['digest'][:10]}.{suffix}")
    return stream.getvalue()


def summarize_result(record):
    metrics = {
        "color": ("median_lab", "swatch", "delta_e_from_tile_median"),
        "dots": ("dot_count", "median_area_px", "ink_area_percent", "unresolved_ink_percent"),
        "weave": ("axis_a_pitch", "axis_b_pitch", "crossing_angle"),
        "coverage": ("percent", "measured"),
        "defects": ("total_observations", "reported_observations"),
        "glare": ("strong_glare_percent", "hotspot_count"),
        "texture": ("texture_orientation", "detail_noise"),
        "craze": ("network_length", "length_unit", "closed_cell_count"),
        "rim": ("all_edges", "rim_band_pixels"),
        "profile": ("line_length", "distance_unit", "supported_stations"),
        "reference": ("reference_title", "delta_e76"),
        "direction": ("left_strong_percent", "right_strong_percent"),
        "marking": ("observed_percent", "midrib_percent_of_observed", "stations"),
        "joint": ("measured_stations", "unmeasured_stations", "median_absolute_step"),
        "blade": ("axis_length", "maximum_width", "observed_stations"),
        "aggregate": ("candidate_clusters", "observed_pixels", "bright_fraction_of_support"),
    }[record["kind"]]
    summary = {key: record[key] for key in (
        "id", "kind", "label", "tile_id", "source_version", "created_at", "version", "basis",
    )}
    summary["tile_name"] = record["provenance"]["specimen"]["name"]
    summary["brief"] = {key: record["metrics"][key] for key in metrics}
    summary["overlay_url"] = f"/api/results/{record['id']}/image/overlay"
    return summary


def summarize_evidence(record):
    keys = ("title", "notes", "id", "tags", "created_at", "tile_id", "tile_name", "tile_version_at_pin", "view")
    return {**{key: record[key] for key in keys}, "count": len(record["measurements"]), "sheet_count": len(record["sheets"])}
