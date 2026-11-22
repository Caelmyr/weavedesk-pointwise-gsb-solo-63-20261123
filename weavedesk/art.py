from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .photos import linear_to_srgb, png_bytes, uint8_image


SEQUENTIAL = (
    (0.00, (23, 50, 62)),
    (0.22, (33, 115, 131)),
    (0.48, (114, 177, 151)),
    (0.74, (235, 204, 114)),
    (1.00, (237, 116, 74)),
)
SIGNED = (
    (0.00, (64, 133, 206)),
    (0.31, (126, 187, 218)),
    (0.50, (52, 57, 62)),
    (0.69, (224, 166, 123)),
    (1.00, (238, 98, 68)),
)
COVERAGE_PALETTE = np.asarray(((235, 178, 94), (218, 216, 146), (91, 168, 143)), dtype=np.float32) / 255
MAX_PREVIEW = 1440
INK = (255, 245, 225)
ACCENT = (246, 139, 96)
SHADOW = (30, 38, 43)


def continuous_palette(values, stops):
    boundaries = [ratio for ratio, _ in stops]
    channels = np.asarray([color for _, color in stops], dtype=np.float32)
    result = np.zeros(values.shape + (3,), dtype=np.float32)
    for channel in range(3):
        result[:, :, channel] = np.interp(values, boundaries, channels[:, channel]) / 255
    return result


def base_rgba(frame, linear=None):
    working = frame.display() if linear is None else linear_to_srgb(linear)
    image = uint8_image(working, frame.valid.astype(np.float32))
    return image.convert("RGBA")


def composite_color(frame, values, valid, stops, *, strength=0.72, unknown=None, source=None):
    underlying = frame.display() if source is None else linear_to_srgb(source)
    base = np.maximum(underlying * 0.57, 0.055)
    foreground = continuous_palette(values, stops)
    base[valid] = (base[valid] * (1 - strength)) + foreground[valid] * strength
    if unknown is not None and np.any(unknown):
        yy, xx = np.ogrid[:valid.shape[0], :valid.shape[1]]
        stripe = ((xx + yy) % 14 <= 3) & unknown
        base[unknown] = (0.22, 0.23, 0.23)
        base[stripe] = (0.64, 0.61, 0.56)
    return uint8_image(base, frame.valid.astype(np.float32))


def magnitude_overlay(frame, values, valid, scale, *, source=None, unknown=None):
    normalized = np.clip(values / max(float(scale), 0.001), 0, 1)
    return composite_color(frame, normalized, valid, SEQUENTIAL, source=source, unknown=unknown)


def signed_overlay(frame, values, valid, scale, *, source=None, unknown=None):
    factor = max(float(scale), 0.001)
    normalized = np.clip(0.5 + values / (2 * factor), 0, 1)
    return composite_color(frame, normalized, valid, SIGNED, source=source, unknown=unknown)


def coverage_overlay(frame, states, requested):
    underlying = frame.display()
    displayed = underlying * 0.48
    for label, color in enumerate(COVERAGE_PALETTE):
        selector = states == label
        displayed[selector] = 0.22 * underlying[selector] + 0.78 * color
    unknown = frame.valid & requested & (states < 0)
    rr, cc = np.ogrid[:states.shape[0], :states.shape[1]]
    dark = ((rr - cc) % 12 < 3) & unknown
    displayed[dark] = (0.12, 0.12, 0.12)
    return uint8_image(displayed, frame.valid.astype(np.float32))


def curve_points(vertices, extent):
    width, height = extent
    return [(round(x * width), round(y * height)) for x, y in vertices]


def reviewed_markers(frame, observations, decisions=None, source=None):
    image = base_rgba(frame, source)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    short_edge = min(width, height)
    line = max(1, round(short_edge / 320))
    radius = max(5, round(short_edge / 92))
    lookup = decisions or {}
    for observation in observations:
        x, y = curve_points([observation["normalized_center"]], image.size)[0]
        status = lookup.get(observation["key"], {}).get("decision")
        if status == "dismiss":
            continue
        color = (174, 233, 188) if status == "confirm" else ACCENT
        # An outer stroke stays legible on bright and dark surfaces.
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=SHADOW, width=line + 2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line)
        draw.line((x + radius + 2, y, x + radius + 6, y), fill=color, width=line)
        if len(observations) <= 35:
            draw.text((x + radius + 5, y - 7), observation["key"], fill=color, stroke_width=2, stroke_fill=SHADOW)
    return image


def profile_overlay(frame, endpoints, band_pixels, source=None):
    image = base_rgba(frame, source)
    draw = ImageDraw.Draw(image)
    points = curve_points(endpoints, image.size)
    width = max(2, int(round(band_pixels)))
    draw.line(points, fill=SHADOW, width=width + 3)
    draw.line(points, fill=ACCENT, width=width)
    for index, (x, y) in enumerate(points):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=INK, outline=SHADOW, width=2)
        draw.text((x + 7, y - 7), "AB"[index], fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image


def annotated_regions(frame, regions, source=None):
    image = base_rgba(frame, source)
    draw = ImageDraw.Draw(image)
    colors = {"ignore": (236, 121, 125), "bare": (238, 184, 120), "glazed": (120, 231, 185), "neutral": (228, 230, 211)}
    width = max(1, round(min(image.size) / 360))
    for region in regions:
        color = colors.get(region["purpose"], (244, 169, 138))
        points = curve_points(region["vertices"], image.size)
        draw.line([*points, points[0]], width=width + 2, fill=SHADOW, joint="curve")
        draw.line([*points, points[0]], width=width, fill=color, joint="curve")
        x, y = points[0]
        draw.text((x + 5, y + 4), region["name"][:36], fill=color, stroke_width=2, stroke_fill=SHADOW)
    return image


def stored_preview(image, blobs):
    output = image.copy()
    output.thumbnail((MAX_PREVIEW, MAX_PREVIEW), Image.Resampling.LANCZOS)
    return blobs.put(png_bytes(output))


def font(size):
    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def clipped_text(draw, phrase, typeface, available):
    if draw.textlength(phrase, font=typeface) <= available:
        return phrase
    label = phrase
    while label and draw.textlength(label + "…", font=typeface) > available:
        label = label[:-1]
    return label + "…"


def centered_image(target, source, box, shade=(39, 48, 52)):
    left, top, width, height = box
    backing = Image.new("RGB", (width, height), shade)
    source = source.convert("RGBA")
    source.thumbnail((width, height), Image.Resampling.LANCZOS)
    offset = ((width - source.width) // 2, (height - source.height) // 2)
    backing.paste(source, offset, source.getchannel("A"))
    target.paste(backing, (left, top))


def comparison_sheet(cards, title):
    columns = min(3, len(cards))
    rows = (len(cards) + columns - 1) // columns
    gap = 22
    item_width, item_height = 330, 252
    image = Image.new("RGB", (gap + columns * (item_width + gap), 90 + rows * (item_height + 70)), (24, 31, 34))
    draw = ImageDraw.Draw(image)
    header, body, small = font(24), font(15), font(12)
    top_line = clipped_text(draw, title, header, image.width - gap * 2)
    draw.text((gap, 24), top_line, font=header, fill=INK)
    for index, card in enumerate(cards):
        row, col = divmod(index, columns)
        left = gap + col * (item_width + gap)
        top = 68 + row * (item_height + 70)
        centered_image(image, card["image"], (left, top, item_width, item_height))
        label = clipped_text(draw, card["heading"], body, item_width)
        draw.text((left, top + item_height + 8), label, font=body, fill=INK)
        detail = clipped_text(draw, card["description"], small, item_width)
        draw.text((left, top + item_height + 30), detail, font=small, fill=(167, 181, 174))
    return image


def load_artifact_picture(blobstore, digest):
    with Image.open(BytesIO(blobstore.get(digest))) as source:
        return source.convert("RGBA")
