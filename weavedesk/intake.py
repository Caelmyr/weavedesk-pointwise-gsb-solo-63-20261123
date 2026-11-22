from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

from PIL import Image, UnidentifiedImageError

from .evidence import result_entry_names
from .model import ANALYSIS_LABELS, InputError, digest_identity, identity, integer, string
from .photos import bounded_preview, png_bytes


MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
MAX_EXAMINED_BYTES = 72 * 1024 * 1024
MAX_MEMBERS = 130
MAX_MEMBER_BYTES = 24 * 1024 * 1024
MAX_MANIFEST = 6 * 1024 * 1024
MAX_PACKET_PIXELS = 3_000_000


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
        stem = PurePosixPath(key).stem
        short = stem.rsplit("-", 1)[-1]
        if short in entries:
            raise InputError("More than one evidence original uses the same digest prefix.")
        entries[short] = key
    return entries


def read_originals(archive, names, documents):
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
        member_digest(archive, names, key, expected)
        listed[expected] = {
            "digest": expected,
            "bytes": info.file_size,
            "name": string(document.get("name"), "Original name", maximum=140),
            "role": string(document.get("role"), "Lighting role", maximum=32),
        }
    if len(index) != len(listed):
        raise InputError("Some original photos were not declared in the packet.")
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


def import_measurements(store, archive, names, records, specimen):
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
        for layer in ("source", "overlay", "paired"):
            digest = pictures.get(layer)
            if layer == "paired" and digest is None:
                continue
            data = member_digest(archive, names, stem + f"-{layer}.png", digest)
            images[layer] = store.blobs.put(decoded_panel(data, f"{label} {layer}"))
        provenance = trusted_provenance(record, specimen)
        collected.append({
            "id": identifier,
            "label": label,
            "kind": kind,
            "source_version": provenance["source_version"],
            "capture": provenance,
            "pictures": images,
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
    original_images = read_originals(archive, entries, manifest.get("originals", []))
    measurements = import_measurements(store, archive, entries, manifest.get("measurements"), tile_id)
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
