from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from PIL.ExifTags import Base as Exif

from .model import InputError, MAX_PIXELS, MAX_UPLOAD


PHOTO_FORMATS = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
}
EXIF_DETAIL = {
    Exif.Make: "camera_make",
    Exif.Model: "camera_model",
    Exif.ExposureTime: "exposure_seconds",
    Exif.ApertureValue: "aperture_apex",
    Exif.FNumber: "f_number",
    Exif.ISOSpeedRatings: "iso",
    Exif.FocalLength: "focal_mm",
    Exif.DateTimeOriginal: "captured_at",
    Exif.LensModel: "lens",
    Exif.Orientation: "orientation",
}
SRGB_ICC = ImageCms.createProfile("sRGB")
PHOTO_PREVIEW = 1280
THUMBNAIL_EDGE = 480
HISTOGRAM_BINS = 32
WHITE_CLIP = 253
BLACK_CLIP = 5


@dataclass(frozen=True)
class DecodedPhoto:
    image: Image.Image
    format: str
    original_mode: str
    original_size: tuple
    color_space: str
    warning: str
    exif: dict
    original_alpha: bool

    @property
    def width(self):
        return self.image.width

    @property
    def height(self):
        return self.image.height


def exif_value(value):
    if isinstance(value, str):
        return value.strip()[:160] or None
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    return round(number, 8) if np.isfinite(number) else None


def small_exif(image):
    source = image.getexif()
    try:
        extension = source.get_ifd(Exif.ExifOffset)
    except (AttributeError, ValueError, TypeError, OSError):
        extension = {}
    exif = {}
    for key, title in EXIF_DETAIL.items():
        if key in extension:
            value = exif_value(extension[key])
        elif key in source:
            value = exif_value(source[key])
        else:
            continue
        if value is not None:
            exif[title] = value
    return exif


def decode_capture(data):
    if not data or len(data) > MAX_UPLOAD:
        raise InputError("Upload a nonempty photo of no more than 24 MiB.")
    try:
        with Image.open(BytesIO(data)) as source:
            source_format = source.format
            if source_format not in PHOTO_FORMATS:
                raise InputError("Upload a PNG, JPEG, WebP, or single-frame TIFF.")
            if getattr(source, "n_frames", 1) != 1:
                raise InputError("Choose one image rather than an animation or multi-page image.")
            width, height = source.size
            if min(width, height) < 24 or width * height > MAX_PIXELS:
                raise InputError("Images must be at least 24 pixels per side and at most 12 megapixels.")
            original_mode = source.mode
            exif = small_exif(source)
            profile = source.info.get("icc_profile")
            alpha_present = "A" in source.getbands() or "transparency" in source.info
            oriented = ImageOps.exif_transpose(source)
            normalized, interpretation, warning = oriented_rgb(oriented, profile, alpha_present)
            return DecodedPhoto(
                image=normalized,
                format=source_format,
                original_mode=original_mode,
                original_size=(width, height),
                color_space=interpretation,
                warning=warning,
                exif=exif,
                original_alpha=alpha_present,
            )
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, IndexError) as error:
        raise InputError("The file could not be decoded as a supported photo.") from error


def normalized_grayscale(image, mode):
    if mode.startswith("I;16"):
        plane = np.asarray(image, dtype=np.float32) / 65535.0
        values = np.round(np.clip(plane * 255, 0, 255)).astype(np.uint8)
        return Image.fromarray(values)
    if mode in ("I", "F"):
        raise InputError("Convert signed or floating-point grayscale captures to unsigned PNG or RGB TIFF.")
    if mode in ("1", "L", "LA"):
        return image.convert("L")
    return None


def oriented_rgb(image, profile, alpha_present):
    alpha = image.convert("RGBA").getchannel("A") if alpha_present else None
    mode = image.mode
    grayscale = normalized_grayscale(image, mode)
    source_space = "assumed sRGB"
    warning = "No embedded color profile. Values use an sRGB assumption."
    if profile:
        try:
            input_profile = ImageCms.getOpenProfile(BytesIO(profile))
            candidate = grayscale if grayscale is not None else image.convert("CMYK" if mode == "CMYK" else "RGB")
            rgb = ImageCms.profileToProfile(candidate, input_profile, SRGB_ICC, outputMode="RGB")
            source_space = "ICC → sRGB"
            warning = ""
        except (ImageCms.PyCMSError, OSError, TypeError, ValueError):
            raise InputError("The embedded color profile could not be converted. Re-export as sRGB.") from None
    else:
        if mode == "CMYK":
            raise InputError("CMYK photos need a compatible embedded profile or an sRGB export.")
        rgb = grayscale.convert("RGB") if grayscale is not None else image.convert("RGB")
    if mode.startswith("I;16"):
        warning = "Native grayscale levels were mapped to 8-bit working intensity before inspection."
    if alpha is not None:
        rgb.putalpha(alpha)
    return rgb, source_space, warning


def png_bytes(image, *, compression=4):
    stream = BytesIO()
    image.save(stream, format="PNG", compress_level=compression)
    return stream.getvalue()


def bounded_preview(image, size):
    result = image.copy()
    result.thumbnail((size, size), Image.Resampling.LANCZOS)
    return result


def safe_name(value, suffix):
    if not value:
        return "capture" + suffix
    clean = Path(value.replace("\\", "/")).name.strip().replace("\x00", "")
    return clean[-140:] or "capture" + suffix


def split_rgba(image):
    values = np.asarray(image.convert("RGBA"), dtype=np.float32)
    rgb = values[:, :, :3] / 255
    alpha = values[:, :, 3] / 255
    return rgb, alpha


def uint8_image(rgb, alpha=None):
    channels = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if alpha is None:
        return Image.fromarray(channels)
    opacity = np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack((channels, opacity)))


def luminance(rgb_linear):
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.sum(rgb_linear * weights, axis=-1, dtype=np.float32)


def srgb_to_linear(values):
    source = np.asarray(values, dtype=np.float32)
    return np.where(
        source <= 0.04045,
        source / 12.92,
        ((source + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(values):
    source = np.maximum(np.asarray(values, dtype=np.float32), 0)
    return np.where(
        source <= 0.0031308,
        source * 12.92,
        1.055 * np.power(source, 1 / 2.4) - 0.055,
    )


def thumbnail_stats(image):
    preview = bounded_preview(image, THUMBNAIL_EDGE).convert("RGBA")
    rgba = np.asarray(preview, dtype=np.uint8)
    support = rgba[:, :, 3] >= 254
    if np.count_nonzero(support) < 16:
        raise InputError("The image has too few opaque pixels for inspection.")
    codes = rgba[:, :, :3][support]
    luma = luminance(srgb_to_linear(codes / 255))
    quantiles = np.quantile(luma, [0.01, 0.25, 0.50, 0.75, 0.99])
    means = np.mean(codes, axis=0)
    histograms = []
    for channel in codes.T:
        counts = np.bincount(channel // (256 // HISTOGRAM_BINS), minlength=HISTOGRAM_BINS)
        histograms.append(counts.tolist())

    grayscale = luminance(srgb_to_linear(rgba[:, :, :3] / 255))
    horizontal_delta = np.abs(grayscale[:, 1:] - grayscale[:, :-1])
    vertical_delta = np.abs(grayscale[1:, :] - grayscale[:-1, :])
    horizontal_valid = support[:, 1:] & support[:, :-1]
    vertical_valid = support[1:, :] & support[:-1, :]
    dx = horizontal_delta[horizontal_valid]
    dy = vertical_delta[vertical_valid]
    edges = np.r_[dx, dy]
    return {
        "supported_percent": round(100 * support.mean(), 2),
        "white_clip_percent": round(100 * np.any(codes >= WHITE_CLIP, axis=1).mean(), 2),
        "black_clip_percent": round(100 * np.all(codes <= BLACK_CLIP, axis=1).mean(), 2),
        "lightness_quantiles": [round(float(value), 4) for value in quantiles],
        "edge_contrast_p90": round(float(np.quantile(edges, 0.90)), 4) if len(edges) else None,
        "rgb_mean": np.rint(means).astype(int).tolist(),
        "histograms": histograms,
        "estimated_from": list(preview.size),
    }


def store_capture(blobs, data, role, filename, note, new_id, *, synthetic=False):
    photo = decode_capture(data)
    stats = thumbnail_stats(photo.image)
    source_digest = blobs.put(data)
    full_digest = blobs.put(png_bytes(photo.image))
    preview_digest = blobs.put(png_bytes(bounded_preview(photo.image, PHOTO_PREVIEW)))
    thumbnail_digest = blobs.put(png_bytes(bounded_preview(photo.image, THUMBNAIL_EDGE)))
    mime = PHOTO_FORMATS[photo.format]
    return {
        "id": new_id,
        "role": role,
        "name": safe_name(filename, "." + photo.format.lower()),
        "note": note,
        "format": photo.format,
        "original_mime": mime,
        "original_mode": photo.original_mode,
        "original_size": list(photo.original_size),
        "width": photo.width,
        "height": photo.height,
        "has_alpha": photo.original_alpha,
        "working_color_space": photo.color_space,
        "warning": photo.warning,
        "exif": photo.exif,
        "bytes": len(data),
        "source_digest": source_digest,
        "working_digest": full_digest,
        "preview_digest": preview_digest,
        "thumbnail_digest": thumbnail_digest,
        "stats": stats,
        "synthetic": synthetic,
    }


def open_working(blobs, photo):
    with Image.open(blobs.require(photo["working_digest"])) as image:
        return split_rgba(image)


def diagnostic_flags(photo):
    stats = photo["stats"]
    flags = []
    if stats["supported_percent"] < 96:
        flags.append("Transparency excludes part of the image.")
    if stats["white_clip_percent"] > 2:
        flags.append("Some highlights reach the capture ceiling.")
    if stats["black_clip_percent"] > 2:
        flags.append("Some shadows are clipped near display black.")
    sharpness = stats["edge_contrast_p90"]
    if sharpness is not None and sharpness < 0.015:
        flags.append("Low edge contrast at preview scale. Glaze texture or defocus may account for this.")
    if photo["warning"]:
        flags.append(photo["warning"])
    return flags
