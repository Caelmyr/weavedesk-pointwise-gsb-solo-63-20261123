import math
import re
from datetime import datetime, timezone


class InputError(ValueError):
    """An invalid capture, measurement, or spec sheet."""


class MissingError(LookupError):
    """The requested record or resource could not be found."""


class ConflictError(RuntimeError):
    """A write would overwrite a newer version or protected resource."""


MAX_PIXELS = 12_000_000
MAX_UPLOAD = 24 * 1024 * 1024
MAX_TILE_IMAGES = 10
MAX_REGIONS = 24
MAX_REGISTERED_DIM = 1800

ID_RE = re.compile(r"^[0-9a-f]{24}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
ROLE_LABELS = {
    "diffuse": "Diffuse front",
    "cross": "Cross-polarized",
    "parallel": "Parallel-polarized",
    "rake-left": "Raking light from left",
    "rake-right": "Raking light from right",
    "reference": "Reference capture",
}
REGION_PURPOSES = {
    "inspection": "Inspection area",
    "bare": "Bare mounting cloth sample",
    "glazed": "Bound textile sample",
    "neutral": "Neutral color card",
    "ignore": "Exclude from measurement",
}
ANALYSIS_LABELS = {
    "dots": "Yarn knot census",
    "weave": "Yarn spacing and angle",
    "color": "Color variation",
    "coverage": "Fabric coverage",
    "defects": "Surface observations",
    "glare": "Polarized glare",
    "texture": "Surface texture",
    "profile": "Line scan",
    "craze": "Fine-line network",
    "rim": "Mount margin color",
    "reference": "Reference match",
    "direction": "Raking light comparison",
    "marking": "Edge stitch visibility",
    "joint": "Mount seam scan",
    "blade": "Binding width",
    "aggregate": "Pale fiber flecks",
    "relief": "Pile relief height",
}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def identity(value, label="ID"):
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise InputError(f"{label} must be a 24-character record ID.")
    return value


def digest_identity(value, label="Image digest"):
    if not isinstance(value, str) or not DIGEST_RE.fullmatch(value):
        raise InputError(f"{label} must be a SHA-256 digest.")
    return value


def dictionary(value, label):
    if not isinstance(value, dict):
        raise InputError(f"{label} must be an object.")
    return value


def collection(value, label, *, minimum=0, maximum=1000):
    if not isinstance(value, (list, tuple)) or not minimum <= len(value) <= maximum:
        raise InputError(f"{label} must contain between {minimum} and {maximum} items.")
    return list(value)


def string(value, label, *, maximum=120, required=True):
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise InputError(f"{label} must be text.")
    cleaned = value.strip()
    if required and not cleaned:
        raise InputError(f"Enter {label.lower()}.")
    if len(cleaned) > maximum or "\x00" in cleaned:
        raise InputError(f"{label} exceeds its text limit or contains a null character.")
    return cleaned


def enumeration(value, choices, label):
    if not isinstance(value, str) or value not in choices:
        available = ", ".join(choices)
        raise InputError(f"{label} must be one of {available}.")
    return value


def scalar(value, label, *, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{label} must be a number.")
    try:
        result = float(value)
    except OverflowError as error:
        raise InputError(f"{label} is outside the supported numerical range.") from error
    if not math.isfinite(result):
        raise InputError(f"{label} must be finite.")
    if lower is not None and result < lower:
        raise InputError(f"{label} must be at least {lower}.")
    if upper is not None and result > upper:
        raise InputError(f"{label} must be at most {upper}.")
    return result


def integer(value, label, *, lower=0, upper=1_000_000):
    if type(value) is not int or not lower <= value <= upper:
        raise InputError(f"{label} must be a whole number between {lower} and {upper}.")
    return value


def flag(value, label):
    if type(value) is not bool:
        raise InputError(f"{label} must be true or false.")
    return value


def numeric_pair(value, label, *, lower=None, upper=None):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise InputError(f"{label} needs two coordinates.")
    return [scalar(number, label, lower=lower, upper=upper) for number in value]


def unique_ids(value, label, *, minimum=0, maximum=32):
    items = collection(value, label, minimum=minimum, maximum=maximum)
    result = [identity(item, label) for item in items]
    if len(set(result)) != len(result):
        raise InputError(f"{label} contains repeated records.")
    return result


def tile_fields(payload, existing=None):
    data = dictionary(payload, "Specimen")
    current = existing or {}
    name = string(data.get("name", current.get("name")), "Specimen name")
    return {
        "name": name,
        "batch": string(data.get("batch", current.get("batch")), "Tray", maximum=80, required=False),
        "body": string(data.get("body", current.get("body")), "Fabric group", maximum=100, required=False),
        "glaze": string(data.get("glaze", current.get("glaze")), "Mount condition", maximum=100, required=False),
        "firing": string(data.get("firing", current.get("firing")), "Imaging visit", maximum=100, required=False),
        "notes": string(data.get("notes", current.get("notes")), "Notes", maximum=2000, required=False),
        "width_mm": optional_size(data.get("width_mm", current.get("width_mm")), "Specimen width"),
        "height_mm": optional_size(data.get("height_mm", current.get("height_mm")), "Specimen height"),
    }


def optional_size(value, label):
    if value is None:
        return None
    return scalar(value, label, lower=0.1, upper=100000)


def region_fields(payload):
    data = dictionary(payload, "Region")
    vertices = collection(data.get("vertices"), "Region vertices", minimum=3, maximum=40)
    return {
        "name": string(data.get("name"), "Region name", maximum=80),
        "purpose": enumeration(data.get("purpose", "inspection"), REGION_PURPOSES, "Region purpose"),
        "vertices": [numeric_pair(item, "Vertex", lower=0, upper=1) for item in vertices],
    }


def library_fields(payload):
    data = dictionary(payload, "Reference specimen")
    return {
        "tile_id": identity(data.get("tile_id"), "Specimen ID"),
        "title": string(data.get("title"), "Reference name", maximum=100),
        "family": string(data.get("family"), "Reference series", maximum=100, required=False),
        "notes": string(data.get("notes"), "Reference notes", maximum=1200, required=False),
        "region_id": identity(data["region_id"], "Region ID") if data.get("region_id") else None,
    }


def flag_record_fields(payload):
    data = dictionary(payload, "Defect review")
    return {
        "decision": enumeration(data.get("decision"), ("confirm", "dismiss", "review"), "Decision"),
        "notes": string(data.get("notes"), "Observation notes", maximum=240, required=False),
    }


def tags(value, *, maximum=10):
    values = collection(value, "Tags", maximum=maximum)
    ordered = {}
    for tag in values:
        label = string(tag, "Tag", maximum=50)
        ordered.setdefault(label.casefold(), label)
    return list(ordered.values())


def match_slot(options, wanted, what):
    for option in options:
        if option["id"] == wanted:
            return option
    raise MissingError(f"The {what} is no longer in this record.")


def model_version(payload):
    data = dictionary(payload, "Request")
    return integer(data.get("version"), "Version", lower=1, upper=2_000_000_000)


def image_roles(runs):
    return {record["role"]: record for record in runs}


def physical_scale(tile, width, height):
    physical_width = tile.get("width_mm")
    physical_height = tile.get("height_mm")
    if physical_width is None or physical_height is None:
        return None
    return physical_width / width, physical_height / height
