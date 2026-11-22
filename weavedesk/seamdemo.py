"""Synthetic broken-ribbon fragment pair for the seam workbench.

The generator renders one continuous indigo-striped woven band, tears it into
a left and a right fragment with ragged edges and frayed warp ends, then
photographs each fragment separately on a dark mount next to a millimetre
ruler.  Both renders share one physical scale, matching the preservation
brief of two same-light, ruler-in-frame photographs.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .photos import linear_to_srgb, png_bytes, srgb_to_linear
from .store import make_id


RIBBON_LENGTH = 1120
RIBBON_WIDTH = 168
WIDTH_PX = 1320
HEIGHT_PX = 540
RIBBON_Y = 196
RULER_HEIGHT = 34
NOTCH_STEP_PX = 30.0
GROUND_TRUTH_NOTCH = 0
WARP_PITCH = 9.0
WEFT_PITCH = 7.0
CLOTH_LINEAR = srgb_to_linear(np.array([(196, 178, 138)], dtype=np.float32) / 255)[0]
INDIGO_LINEAR = srgb_to_linear(np.array([(36, 72, 116)], dtype=np.float32) / 255)[0]
PALE_LINEAR = srgb_to_linear(np.array([(228, 214, 176)], dtype=np.float32) / 255)[0]
MOUNT_LINEAR = srgb_to_linear(np.array([(52, 56, 54)], dtype=np.float32) / 255)[0]


@dataclass(frozen=True)
class TearGeometry:
    split: int
    gap: int
    left_tear: np.ndarray   # tear x per ribbon row for the left fragment
    right_tear: np.ndarray  # tear x per ribbon row for the right fragment
    left_strands: tuple
    right_strands: tuple
    paired_rows: tuple
    lone_left_rows: tuple
    lone_right_rows: tuple


def _tear_geometry(generator):
    split = 556
    rows = np.arange(RIBBON_WIDTH)
    jagged = (
        np.rint(7 * np.sin(rows / 13.0) + 4 * np.sin(rows / 5.3 + 1.7)).astype(int)
        + generator.integers(-3, 4, size=RIBBON_WIDTH)
    )
    left_tear = split + jagged
    right_tear = split + 64 + jagged + generator.integers(-2, 3, size=RIBBON_WIDTH)
    paired_rows = (44, 108)
    lone_left_rows = (78,)
    lone_right_rows = (132,)

    def strand(y, side):
        edge = left_tear[y] if side == "left" else right_tear[y]
        length = int(generator.integers(14, 26))
        drift = int(generator.integers(-2, 3))
        return {"y": y, "edge": int(edge), "length": length, "drift": drift}

    left_strands = tuple(strand(y, "left") for y in (*paired_rows, *lone_left_rows))
    right_strands = tuple(strand(y, "right") for y in (*paired_rows, *lone_right_rows))
    return TearGeometry(
        split=split,
        gap=64,
        left_tear=left_tear,
        right_tear=right_tear,
        left_strands=left_strands,
        right_strands=right_strands,
        paired_rows=paired_rows,
        lone_left_rows=lone_left_rows,
        lone_right_rows=lone_right_rows,
    )


def _weave_texture(shape, generator):
    height, width = shape
    y, x = np.indices(shape, dtype=np.float32)
    warp_phase = (x + 0.35 * np.sin(y / 11.0)) / WARP_PITCH
    weft_phase = (y + 0.25 * np.sin(x / 17.0)) / WEFT_PITCH
    warp_ridge = np.exp(-9.0 * (warp_phase - np.round(warp_phase)) ** 2)
    weft_ridge = np.exp(-8.0 * (weft_phase - np.round(weft_phase)) ** 2)
    interlace = ((np.round(warp_phase).astype(int) + np.round(weft_phase).astype(int)) % 2) == 0
    yarn = np.where(interlace, np.maximum(0.95 * warp_ridge, 0.7 * weft_ridge),
                    np.maximum(0.7 * warp_ridge, 0.95 * weft_ridge))
    noise = generator.normal(0, 0.022, shape).astype(np.float32)
    broad = 0.94 + 0.05 * np.sin(x / 95.0) + 0.03 * np.cos(y / 41.0)
    return np.clip(yarn * broad + noise, 0.6, 1.18).astype(np.float32)


def _blue_bands():
    """Two blue stripe groups straddle the tear; only middle notch aligns both."""
    # (start row, end row) groups in full-ribbon coordinates.
    groups = ((30, 52), (96, 118))
    # A shorter decoy group exists only on the right fragment, tempting the up notch.
    decoy = (8, 18)
    return groups, decoy


def build_ribbon(generator):
    texture = _weave_texture((RIBBON_WIDTH, RIBBON_LENGTH), generator)
    ribbon = np.broadcast_to(CLOTH_LINEAR, (RIBBON_WIDTH, RIBBON_LENGTH, 3)).copy()
    ribbon *= texture[:, :, None]
    groups, decoy = _blue_bands()
    blue_mask = np.zeros((RIBBON_WIDTH, RIBBON_LENGTH), dtype=bool)
    for start, end in groups:
        blue_mask[start:end, :] = True
    # The decoy blue group only appears past the right tear, slightly faded.
    decoy_mask = np.zeros_like(blue_mask)
    decoy_mask[decoy[0]:decoy[1], :] = True
    ribbon[blue_mask] = INDIGO_LINEAR * texture[blue_mask, None]
    ribbon_decoy = ribbon.copy()
    faded_indigo = CLOTH_LINEAR * 0.45 + INDIGO_LINEAR * 0.55
    ribbon_decoy[decoy_mask] = faded_indigo * texture[decoy_mask, None]
    return ribbon, ribbon_decoy, blue_mask, decoy_mask


def _paint_strands(canvas, strands, side, origin_x, origin_y):
    for strand in strands:
        y0 = origin_y + strand["y"]
        x0 = origin_x + strand["edge"]
        points = []
        for step in range(strand["length"]):
            if side == "left":
                x = x0 + step
            else:
                x = x0 - step
            y = y0 + round(strand["drift"] * step / max(1, strand["length"]))
            points.append((x, y))
        for x, y in points:
            if 0 <= y < canvas.shape[0] and 0 <= x < canvas.shape[1]:
                canvas[y, x] = PALE_LINEAR
                if y + 1 < canvas.shape[0]:
                    canvas[y + 1, x] = PALE_LINEAR * 0.92


def _ruler(draw, x0, x1, y0, pixels_per_mm=4.0):
    draw.rectangle((x0, y0, x1, y0 + RULER_HEIGHT), fill=(214, 205, 178), outline=(40, 40, 36))
    span = int((x1 - x0) / pixels_per_mm)
    for mm in range(span + 1):
        x = int(x0 + mm * pixels_per_mm)
        if x > x1:
            break
        tick = 12 if mm % 10 == 0 else 8 if mm % 5 == 0 else 4
        draw.line((x, y0, x, y0 + tick), fill=(35, 35, 30))
        if mm % 10 == 0:
            draw.text((x + 2, y0 + 13), str(mm), fill=(35, 35, 30))


def _render_fragment(ribbon_full, support_ribbon, tear, strands, side, geometry, seed, with_decoy):
    generator = np.random.default_rng(seed)
    canvas = np.broadcast_to(MOUNT_LINEAR, (HEIGHT_PX, WIDTH_PX, 3)).copy()
    canvas *= (0.97 + generator.normal(0, 0.015, (HEIGHT_PX, WIDTH_PX, 1))).astype(np.float32)
    origin_x = 70 if side == "left" else WIDTH_PX - 70 - RIBBON_LENGTH
    origin_y = RIBBON_Y
    rows = np.arange(RIBBON_WIDTH)
    for y in rows:
        edge = int(tear[y])
        if side == "left":
            x_range = np.arange(0, edge)
        else:
            x_range = np.arange(edge, RIBBON_LENGTH)
        target_x = x_range + origin_x
        target_y = np.full_like(target_x, y + origin_y)
        keep = (target_x >= 0) & (target_x < WIDTH_PX)
        canvas[target_y[keep], target_x[keep]] = ribbon_full[y, x_range[keep]]
    _paint_strands(canvas, strands, side, origin_x, origin_y)
    image = Image.fromarray(np.rint(np.clip(linear_to_srgb(canvas), 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    _ruler(draw, origin_x, origin_x + RIBBON_LENGTH, RIBBON_Y + RIBBON_WIDTH + 26)
    draw.text((origin_x, RIBBON_Y - 28), f"Ribbon {side} fragment · mm scale", fill=(228, 226, 210))

    # Outline polygon in normalized photo coordinates, a simple clockwise
    # boundary along the outer long edge, the torn edge, and the far side.
    edge_points = [
        (int(tear[y]) + origin_x, y + origin_y)
        for y in range(0, RIBBON_WIDTH, 8)
    ]
    top_tear = (int(tear[0]) + origin_x, origin_y)
    bottom_tear = (int(tear[RIBBON_WIDTH - 1]) + origin_x, origin_y + RIBBON_WIDTH)
    top_outer = (origin_x, origin_y)
    bottom_outer = (origin_x, origin_y + RIBBON_WIDTH)
    if side == "left":
        polygon = [top_outer, top_tear, *edge_points[1:], bottom_tear, bottom_outer]
    else:
        far_x = origin_x + RIBBON_LENGTH
        top_far = (far_x, origin_y)
        bottom_far = (far_x, origin_y + RIBBON_WIDTH)
        polygon = [top_tear, *edge_points[1:-1], bottom_tear, bottom_far, top_far]
    outline = [[round(x / WIDTH_PX, 6), round(y / HEIGHT_PX, 6)] for x, y in polygon]
    corners = [
        [round(origin_x / WIDTH_PX, 6), round(origin_y / HEIGHT_PX, 6)],
        [round((origin_x + RIBBON_LENGTH) / WIDTH_PX, 6), round(origin_y / HEIGHT_PX, 6)],
        [round((origin_x + RIBBON_LENGTH) / WIDTH_PX, 6), round((origin_y + RIBBON_WIDTH) / HEIGHT_PX, 6)],
        [round(origin_x / WIDTH_PX, 6), round((origin_y + RIBBON_WIDTH) / HEIGHT_PX, 6)],
    ]
    return image, outline, corners, {"origin_x": origin_x, "origin_y": origin_y, "pixels_per_mm": 4.0}


def render_demo_pair(seed=20261122):
    generator = np.random.default_rng(seed)
    geometry = _tear_geometry(generator)
    ribbon, ribbon_decoy, _, _ = build_ribbon(generator)
    left_image, left_outline, left_corners, left_meta = _render_fragment(
        ribbon, None, geometry.left_tear, geometry.left_strands, "left", geometry, seed + 1, False
    )
    right_image, right_outline, right_corners, right_meta = _render_fragment(
        ribbon_decoy, None, geometry.right_tear, geometry.right_strands, "right", geometry, seed + 2, True
    )
    return {
        "left": {"image": left_image, "outline": left_outline, "corners": left_corners, **left_meta},
        "right": {"image": right_image, "outline": right_outline, "corners": right_corners, **right_meta},
        "notch_step_px": NOTCH_STEP_PX,
        "ground_truth_notch": GROUND_TRUTH_NOTCH,
        "paired_rows": geometry.paired_rows,
        "lone_left_rows": geometry.lone_left_rows,
        "lone_right_rows": geometry.lone_right_rows,
        "band_size": [RIBBON_LENGTH, RIBBON_WIDTH],
        "photo_size": [WIDTH_PX, HEIGHT_PX],
    }


def demo_photo_records(store, pair):
    records = {}
    for side in ("left", "right"):
        item = pair[side]
        photo = store_capture_seam(
            store.blobs, png_bytes(item["image"]),
            f"ribbon-{side}-fragment.png", make_id(),
        )
        records[side] = {
            "photo": photo,
            "outline": item["outline"],
            "corners": item["corners"],
            "pixels_per_mm": item["pixels_per_mm"],
        }
    return records


def demo_specimen(store, records):
    """Persist a specimen carrying the two fragment captures."""
    photos = []
    for side in ("left", "right"):
        photo = dict(records[side]["photo"])
        # Keep the exact band corners and torn outline so the studio can
        # rebuild the seam without re-marking the practice images.
        photo["seam_corners"] = records[side]["corners"]
        photo["seam_outline"] = records[side]["outline"]
        photo["pixels_per_mm"] = records[side]["pixels_per_mm"]
        photos.append(photo)
    tile = {
        "name": "Broken indigo ribbon · fragment pair",
        "batch": "Seam practice tray",
        "body": "Woven band with indigo edge stripes",
        "glaze": "Unlined",
        "firing": "Diffuse survey, ruler in frame",
        "notes": "Synthetic broken-ribbon fragments photographed under one light with millimetre rulers.",
        "width_mm": 280.0,
        "height_mm": 42.0,
        "synthetic": True,
        "photos": photos,
    }
    return store.create_tile(tile)


def store_capture_seam(blobs, data, filename, new_id):
    from .photos import store_capture

    return store_capture(blobs, data, "diffuse", filename,
                         "Synthetic broken-ribbon fragment with millimetre ruler.", new_id, synthetic=True)
