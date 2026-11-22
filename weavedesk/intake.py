from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

from PIL import Image, ImageOps, UnidentifiedImageError

from .evidence import result_entry_names
from .model import ANALYSIS_LABELS, InputError, digest_identity, identity, integer, string
from .photos import bounded_preview, png_bytes
from .redaction import solid_cover, validated_quad


MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
MAX_EXAMINED_BYTES = 72 * 1024 * 1024
MAX_MEMBERS = 130
MAX_MEMBER_BYTES = 24 * 1024 * 1024
MAX_MANIFEST = 6 * 1024 * 1024
MAX_PACKET_PIXELS = 3_000_000
MAX_ORIGINAL_PIXELS = 24_000_000
IMAGE_SUFFIXES = ("png", "jpg", "jpeg", "tif", "tiff", "webp")


def archive_key(value):
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or value != str(path)
        or ".." in path.parts
        or len(path.parts) > 4
        or any(len(part) > 125 for part in path.parts)
    ):
        raise InputError("Archive contains a filename outside the evidence packet layout.")
    return value


def archive_index(archive):
    items = archive.infolist()
    if not items or len(items) > MAX_MEMBERS:
        raise InputError("Choose an evidence ZIP with 130 entries or fewer.")
    names = {}
    size = 0
    for info in items:
        name = archive_key(info.filename)
        if name in names:
            raise InputError("The evidence archive has repeated file names.")
        if info.is_dir() or info.flag_bits & 1:
            raise InputError("Evidence archives require regular, readable entries.")
        if info.file_size > MAX_MEMBER_BYTES:
            raise InputError("An evidence entry exceeds the per-entry size limit.")
        size += info.file_size
        if size > MAX_EXAMINED_BYTES:
            raise InputError("The expanded evidence archive exceeds 72 MiB.")
        names[name] = info
    return names


def read_member(archive, names, key, limit=MAX_MEMBER_BYTES):
    if key not in names:
        raise InputError(f"The archive is missing {key}.")
    info = names[key]
    if info.file_size > limit:
        raise InputError(f"{key} exceeds the reader's size limit.")
    with archive.open(info) as stream:
        data = stream.read(limit + 1)
    if len(data) != info.file_size or len(data) > limit:
        raise InputError(f"{key} does not match its declared size.")
    return data


def reject_duplicate_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise InputError("Evidence JSON has repeated property names.")
        value[key] = item
    return value


def reject_nonfinite(value):
    raise InputError(f"Unsupported numerical constant in evidence JSON: {value}.")


def archive_json(data, title):
    try:
        record = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_nonfinite,
            object_pairs_hook=reject_duplicate_fields,
        )
    except (UnicodeError, ValueError, RecursionError) as error:
        raise InputError(f"{title} is not a readable evidence record.") from error
    if not isinstance(record, dict):
        raise InputError(f"{title} must contain an object.")
    return record


def member_digest(archive, names, key, claimed):
    digest_identity(claimed)
    data = read_member(archive, names, key)
    actual = sha256(data).hexdigest()
    if actual != claimed:
        raise InputError(f"{key} does not match the packet's SHA-256 digest.")
    return data


def decoded_panel(data, label):
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise InputError(f"{label} must be a single PNG image.")
            if min(image.size) < 12 or image.width * image.height > MAX_PACKET_PIXELS:
                raise InputError(f"{label} falls outside the evidence display size.")
            panel = bounded_preview(image.convert("RGBA"), 1280)
            return png_bytes(panel)
    except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise InputError(f"{label} cannot be decoded.") from error


def original_index(names):
    entries = {}
    for key in names:
        if not key.startswith("originals/"):
            continue
        suffix = PurePosixPath(key).suffix.lower().lstrip(".")
        if suffix not in IMAGE_SUFFIXES:
            continue
        stem = PurePosixPath(key).stem
        short = stem.rsplit("-", 1)[-1]
        if short in entries:
            raise InputError("More than one evidence original uses the same digest prefix.")
        entries[short] = key
    return entries


def decoded_original(data, label):
    """Store a bounded display copy of an attached original without altering it."""
    try:
        with Image.open(BytesIO(data)) as source:
            source_format = source.format
            if source_format not in ("PNG", "JPEG", "TIFF", "WEBP") or getattr(source, "n_frames", 1) != 1:
                raise InputError(f"{label} must be a single-frame PNG, JPEG, TIFF or WebP image.")
            if min(source.size) < 24 or source.width * source.height > MAX_ORIGINAL_PIXELS:
                raise InputError(f"{label} falls outside the evidence display size.")
            upright = ImageOps.exif_transpose(source).convert("RGBA")
            panel = bounded_preview(upright, 1280)
            return png_bytes(panel)
    except InputError:
        raise
    except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise InputError(f"{label} cannot be decoded.") from error


def declared_cover(document):
    if document is None:
        return None
    if not isinstance(document, dict):
        raise InputError("The packet cover description is incomplete.")
    if document.get("applied") is False:
        return None
    unknown = set(document) - {
        "applied", "shape", "continuous", "width_mm", "height_mm", "angle_deg", "center",
        "plane_size", "plane_corners", "plane_fraction", "fill_rgb", "members",
    }
    if unknown:
        raise InputError("The packet declares more than one cover size or position.")
    if document.get("continuous") is not True or document.get("shape") != "rectangle":
        raise InputError("The packet cover must be one continuous rectangle.")
    width = document.get("width_mm")
    height = document.get("height_mm")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise InputError("The packet cover lacks its physical size.")
    measured = sorted((float(width), float(height)))
    if abs(measured[0] - 18) > 1e-6 or abs(measured[1] - 22) > 1e-6:
        raise InputError("The packet cover must measure 18 × 22 mm.")
    center = document.get("center")
    if not isinstance(center, list) or len(center) != 2:
        raise InputError("The packet cover lacks one centre position.")
    for value in center:
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise InputError("The packet cover centre must lie on the registered plane.")
    corners = document.get("plane_corners")
    if not isinstance(corners, list) or len(corners) != 4:
        raise InputError("The packet cover lacks its registered-plane corners.")
    quad = []
    for point in corners:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(value, (int, float)) and 0 <= float(value) <= 1 for value in point)
        ):
            raise InputError("The packet cover corners must lie on the registered plane.")
        quad.append([round(float(point[0]), 9), round(float(point[1]), 9)])
    members = document.get("members")
    if members is not None and not isinstance(members, dict):
        raise InputError("The packet cover member list is invalid.")
    return {
        "width_mm": round(float(width), 6),
        "height_mm": round(float(height), 6),
        "angle_deg": float(document.get("angle_deg", 0)),
        "center": [round(float(center[0]), 9), round(float(center[1]), 9)],
        "plane_corners": quad,
        "fill_rgb": document.get("fill_rgb") if isinstance(document.get("fill_rgb"), list) else [27, 32, 34],
        "members": members or {},
    }


def cover_receipt(archive, names, key, expected, cover):
    receipt = archive_json(read_member(archive, names, key, MAX_MANIFEST), "Cover receipt")
    if receipt.get("kind") != "evidence-cover-receipt":
        raise InputError("A covered original lacks its cover receipt.")
    if receipt.get("covered_digest") != expected["digest"] or receipt.get("original_digest") != expected["original_digest"]:
        raise InputError("A cover receipt does not describe the original it accompanies.")
    quad = validated_quad(receipt.get("cover", {}).get("photo_quad"), "Cover position")
    declared = receipt.get("cover", {})
    for key_name in ("width_mm", "height_mm"):
        if declared.get(key_name) != cover[key_name]:
            raise InputError("A cover receipt disagrees with the packet's one cover size.")
    if declared.get("center") != cover["center"] or declared.get("continuous") is not True:
        raise InputError("A cover receipt disagrees with the packet's one cover position.")
    if receipt.get("photo_size") != expected.get("upright_size"):
        raise InputError("A cover receipt does not match the covered original's size.")
    return quad


def read_originals(store, archive, names, documents, cover):
    if not isinstance(documents, list) or len(documents) > 24:
        raise InputError("Evidence contains too many original captures.")
    index = original_index(names)
    listed = {}
    for document in documents:
        if not isinstance(document, dict):
            raise InputError("An original capture descriptor is incomplete.")
        expected = digest_identity(document.get("digest"))
        key = index.get(expected[:10])
        if key is None or expected in listed:
            raise InputError("An original capture does not have a unique archive entry.")
        info = names[key]
        data = member_digest(archive, names, key, expected)
        stored = decoded_original(data, f"Original capture {document.get('name', '')}")
        covered = document.get("covered") is True
        original_digest = document.get("original_digest")
        if cover is None and covered:
            raise InputError("The archive marks a covered original but declares no packet cover.")
        if cover is not None and not covered:
            raise InputError("A covered packet cannot carry an original without the cover applied.")
        entry = {
            "digest": expected,
            "preview_digest": store.blobs.put(stored),
            "bytes": len(data),
            "name": string(document.get("name"), "Original name", maximum=140),
            "role": string(document.get("role"), "Lighting role", maximum=32),
            "covered": covered,
            "original_digest": None,
            "cover_quad": None,
            "upright_size": document.get("upright_size"),
        }
        if covered:
            entry["original_digest"] = digest_identity(original_digest, "Original photograph digest")
            receipt_key = document.get("cover_quad_name")
            if not isinstance(receipt_key, str):
                raise InputError("A covered original lacks its cover receipt file.")
            archive_key(receipt_key)
            if not isinstance(entry["upright_size"], list) or len(entry["upright_size"]) != 2:
                raise InputError("A covered original does not declare its picture size.")
            quad = cover_receipt(archive, names, receipt_key, entry, cover)
            entry["cover_quad"] = quad
            member = cover["members"].get(expected)
            if not isinstance(member, dict) or member.get("quad") != quad:
                raise InputError("The packet cover does not match a covered original's receipt.")
            if solid_cover(Image.open(BytesIO(data)), quad, cover["fill_rgb"]) < 0.98:
                raise InputError("A covered original does not actually show the declared cover.")
        listed[expected] = entry
    if len(index) != len(listed):
        raise InputError("Some original photos were not declared in the packet.")
    if cover is not None:
        undocumented = set(cover["members"]) - set(listed)
        if undocumented:
            raise InputError("The packet cover names an original that is not declared.")
    return list(listed.values())


def measurement_descriptor(entry, expected_tile):
    if not isinstance(entry, dict):
        raise InputError("Evidence has an invalid measurement descriptor.")
    record_id = identity(entry.get("id"), "Measurement ID")
    if entry.get("tile_id") != expected_tile:
        raise InputError("An evidence measurement belongs to a different specimen.")
    kind = entry.get("kind")
    label = entry.get("label")
    if not isinstance(kind, str) or kind not in ANALYSIS_LABELS or ANALYSIS_LABELS[kind] != label:
        raise InputError("Evidence measurement type or label is unsupported.")
    return record_id, kind, label


def trusted_provenance(record, expected_id):
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("specimen"), dict):
        raise InputError("A measurement lacks specimen provenance.")
    source = provenance["specimen"]
    if source.get("id") != expected_id:
        raise InputError("Measurement provenance points to a different specimen.")
    photo = provenance.get("photo")
    if not isinstance(photo, dict):
        raise InputError("A measurement lacks capture provenance.")
    return {
        "name": string(photo.get("display_name"), "Capture name", maximum=140),
        "role": string(photo.get("role"), "Capture role", maximum=32),
        "source_digest": digest_identity(photo.get("original_digest"), "Original photograph digest"),
        "working_digest": digest_identity(photo.get("working_digest"), "Working image digest"),
        "source_version": integer(record.get("source_version"), "Source revision", lower=1, upper=2_000_000_000),
    }


def declared_measurement_cover(document, sizes, cover):
    """A plane-position cover recorded on a measurement's exported pictures."""
    if document is None:
        if cover is not None:
            raise InputError("A covered packet must record the cover on every measurement.")
        return None
    if not isinstance(document, dict):
        raise InputError("A measurement's cover description is invalid.")
    corners = document.get("plane_corners")
    if not isinstance(corners, list) or len(corners) != 4:
        raise InputError("A measurement cover lacks its registered-plane corners.")
    quad = []
    for point in corners:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(value, (int, float)) and 0 <= float(value) <= 1 for value in point)
        ):
            raise InputError("A measurement cover corner lies outside the registered plane.")
        quad.append([round(float(point[0]), 9), round(float(point[1]), 9)])
    if cover is None:
        raise InputError("A measurement declares a cover the packet itself does not apply.")
    if quad != cover["plane_corners"]:
        raise InputError("A measurement's cover differs from the packet's one cover position.")
    if document.get("width_mm") != cover["width_mm"] or document.get("height_mm") != cover["height_mm"]:
        raise InputError("A measurement's cover differs from the packet's one cover size.")
    for layer, size in (document.get("layers") or {}).items():
        if sizes is None:
            continue
        if layer not in sizes:
            raise InputError("A measurement cover names a picture the packet does not carry.")
        if not isinstance(size, list) or len(size) != 2 or list(size) != sizes[layer]:
            raise InputError("A measurement cover does not match its exported picture size.")
    return {
        "plane_corners": quad,
        "width_mm": cover["width_mm"],
        "height_mm": cover["height_mm"],
        "fill_rgb": cover["fill_rgb"],
    }


def import_measurements(store, archive, names, records, specimen, cover):
    if not isinstance(records, list) or not 1 <= len(records) <= 12:
        raise InputError("Evidence needs between 1 and 12 measurement records.")
    collected = []
    seen = set()
    for record in records:
        identifier, kind, label = measurement_descriptor(record, specimen)
        if identifier in seen:
            raise InputError("Evidence repeats a measurement.")
        seen.add(identifier)
        stem = "measurements/" + result_entry_names(record)
        receipt = archive_json(read_member(archive, names, stem + ".json", MAX_MANIFEST), label)
        if record != receipt:
            raise InputError(f"The separate {label} record differs from the manifest.")
        pictures = record.get("pictures")
        if not isinstance(pictures, dict):
            raise InputError("Measurement picture manifest is missing.")
        images = {}
        sizes = {}
        declared = declared_measurement_cover(record.get("cover"), None, cover)
        for layer in ("source", "overlay", "paired"):
            digest = pictures.get(layer)
            if layer == "paired" and digest is None:
                continue
            data = member_digest(archive, names, stem + f"-{layer}.png", digest)
            images[layer] = store.blobs.put(decoded_panel(data, f"{label} {layer}"))
            with Image.open(BytesIO(data)) as picture:
                sizes[layer] = list(picture.size)
                if declared and solid_cover(picture, declared["plane_corners"], declared["fill_rgb"]) < 0.98:
                    raise InputError(f"The covered {label} {layer} picture does not actually show the declared cover.")
        cover_record = declared_measurement_cover(record.get("cover"), sizes, cover)
        provenance = trusted_provenance(record, specimen)
        collected.append({
            "id": identifier,
            "label": label,
            "kind": kind,
            "source_version": provenance["source_version"],
            "capture": provenance,
            "pictures": images,
            "cover": cover_record,
        })
    return collected


def import_sheets(store, archive, names, declarations):
    if not isinstance(declarations, list) or not 1 <= len(declarations) <= 2:
        raise InputError("Evidence needs one or two sheet digests.")
    sanitized = []
    for index, expected in enumerate(declarations, 1):
        key = f"sheets/sheet-{index:02d}.png"
        data = member_digest(archive, names, key, expected)
        sanitized.append(store.blobs.put(decoded_panel(data, f"Inspection sheet {index}")))
    return sanitized


def viewable_packet(store, archive):
    entries = archive_index(archive)
    manifest = archive_json(read_member(archive, entries, "manifest.json", MAX_MANIFEST), "Manifest")
    source_id = identity(manifest.get("id"), "Evidence ID")
    tile_id = identity(manifest.get("tile_id"), "Specimen ID")
    caption = string(manifest.get("title"), "Evidence title", maximum=115)
    notes = string(manifest.get("notes"), "Evidence notes", maximum=2400, required=False)
    cover = declared_cover(manifest.get("redaction"))
    original_images = read_originals(store, archive, entries, manifest.get("originals", []), cover)
    if cover is not None and not original_images:
        raise InputError("A covered packet must attach at least one covered original.")
    measurements = import_measurements(store, archive, entries, manifest.get("measurements"), tile_id, cover)
    sheets = import_sheets(store, archive, entries, manifest.get("sheets"))
    revision = integer(manifest.get("tile_version_at_pin"), "Specimen revision", lower=1, upper=2_000_000_000)
    return {
        "source_evidence_id": source_id,
        "external_tile_id": tile_id,
        "title": caption,
        "notes": notes,
        "specimen": string(manifest.get("tile_name"), "Specimen name"),
        "specimen_revision": revision,
        "originals": original_images,
        "measurements": measurements,
        "sheets": sheets,
        "redaction": cover,
        "report_date": string(manifest.get("created_at"), "Recorded date", maximum=100),
        "synthetic": manifest.get("synthetic") is True,
    }


def inspect_packet(store, raw):
    if not raw or len(raw) > MAX_ARCHIVE_BYTES:
        raise InputError("Choose a nonempty WeaveDesk evidence ZIP of at most 24 MiB.")
    digest = sha256(raw).hexdigest()
    existing = store.received_by_digest(digest)
    if existing:
        return existing, False
    try:
        with ZipFile(BytesIO(raw)) as archive:
            record = viewable_packet(store, archive)
    except (BadZipFile, NotImplementedError, OSError, RuntimeError) as error:
        raise InputError("The file could not be read as a WeaveDesk evidence ZIP.") from error
    record["archive_digest"] = digest
    record["packed_bytes"] = len(raw)
    return store.save_received(record)
