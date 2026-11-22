"""Verify unresolved capture positions stay blank instead of borrowing peaks."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from scipy import ndimage

from weavedesk.geometry import DEFAULT_CORNERS, homography, specimen_frame
from weavedesk.demo import load_training
from weavedesk.store import Store
from weavedesk.survey import repeat_spacing
from weavedesk.photos import png_bytes, srgb_to_linear, linear_to_srgb, open_working


def main():
    data = Path(tempfile.mkdtemp())
    store = Store(data)
    tiles, _ = load_training(store)
    tile = dict(tiles[0])
    base_photo = next(p for p in tile["photos"] if p["role"] == "diffuse")
    region = next(r for r in tile["regions"] if r["name"] == "Visible textile field")

    # Build a re-registration: the new capture is the same plane but with a
    # heavy local blur across the lower half of the inspection region.
    frame = specimen_frame(store.blobs, tile, base_photo["id"])
    width, height = frame.extent
    rgb = linear_to_srgb(frame.linear)
    vertices = np.asarray(region["vertices"])
    top = int(vertices[:, 1].min() * height)
    bottom = int(vertices[:, 1].max() * height)
    cut = top + (bottom - top) // 2
    # The lower half is replaced by a constant mid-tone: exposed and in focus,
    # but with no resolvable yarn periodicity at all.
    blurred_plane = rgb.copy()
    blurred_plane[cut:] = np.asarray((0.42, 0.39, 0.36), dtype=np.float32)

    # Render the edited plane back into the original capture framing so the
    # normal registration warp recovers it pixel-for-pixel.
    source_rgb, source_alpha = open_working(store.blobs, base_photo)
    photo_h, photo_w = source_alpha.shape
    inverse = np.linalg.inv(homography(DEFAULT_CORNERS, tile["plane"]["corners"]))
    py, px = np.indices((photo_h, photo_w), dtype=np.float32)
    u = (px + 0.5) / photo_w
    v = (py + 0.5) / photo_h
    denominator = inverse[2, 0] * u + inverse[2, 1] * v + inverse[2, 2]
    map_x = (inverse[0, 0] * u + inverse[0, 1] * v + inverse[0, 2]) / denominator * width - 0.5
    map_y = (inverse[1, 0] * u + inverse[1, 1] * v + inverse[1, 2]) / denominator * height - 0.5
    coords = (map_y, map_x)
    warped = np.stack([
        ndimage.map_coordinates(blurred_plane[..., channel], coords, order=1, mode="nearest")
        for channel in range(3)
    ], axis=2)
    # The working capture is an opaque photograph; any pixel outside the mapped
    # plane is filled with a mid-tone so it cannot trip highlight/shadow clipping.
    fill = np.asarray((0.45, 0.42, 0.4), dtype=np.float32)
    inside = ndimage.map_coordinates(frame.valid.astype(np.float32), coords,
                                     order=1, mode="constant", cval=0)[..., None]
    warped = warped * inside + fill * (1.0 - inside)
    image = Image.fromarray(np.rint(np.clip(warped, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    blurred_id = "0123456789abcdef01234567"
    record = {
        **base_photo,
        "id": blurred_id,
        "name": "blurred-repeat.png",
        "source_digest": store.blobs.put(png_bytes(image)),
        "working_digest": store.blobs.put(png_bytes(image)),
        "preview_digest": store.blobs.put(png_bytes(image)),
        "thumbnail_digest": store.blobs.put(png_bytes(image)),
        "synthetic": True,
    }
    tile["photos"] = [*tile["photos"], record]
    tile["registrations"][blurred_id] = {
        "base_photo_id": base_photo["id"],
        "corners": tile["plane"]["corners"],
        "support_percent": 100.0,
    }
    tile["survey_plan"] = {
        "survey_axis_deg": 90.0,
        "allowed_axes_deg": [0.0, 90.0],
        "pins": [],
        "region_ids": [region["id"]],
    }

    from weavedesk.geometry import specimen_frame as _frame
    frame_b = _frame(store.blobs, tile, blurred_id)
    print("frame_b valid mean", float(frame_b.valid.mean()), "extent", frame_b.extent,
          "clipped", float(frame_b.clipped.mean()))

    metrics, _ = repeat_spacing(store, tile, tile["survey_plan"], base_photo["id"], blurred_id, 42.0, 14.0)
    segments = metrics["segments"]
    assert segments, "segments should be produced"
    from collections import Counter
    print("total", len(segments), "statuses", Counter(s["status"] for s in segments))
    centers = sorted(s["center_xy"][1] for s in segments)
    midpoint = cut / height
    print("split at cut y", midpoint)
    blurry = [s for s in segments if s["center_xy"][1] >= midpoint]
    sharp = [s for s in segments if s["center_xy"][1] < midpoint]
    print("sharp statuses", Counter(s["status"] for s in sharp))
    print("blur statuses", Counter(s["status"] for s in blurry))
    print("blur reasons", Counter(s["blank_reason"] for s in blurry))
    assert blurry and sharp, "expected segments on both halves"
    for segment in blurry:
        assert segment["pitch_b"] is None, "blurred side must not yield a fabricated later pitch"
        assert segment["status"] == "blank", "blurred side segments stay blank"
    assert any(s["status"] == "measured" for s in sharp), "sharp side still measures"
    measured = [s for s in sharp if s["status"] == "measured"]
    for segment in measured:
        assert segment["pitch_a"] > 0 and segment["pitch_b"] > 0
    print(f"ok - {len(blurry)} blurred segments blank, {len(measured)} sharp segments measured")
    print("blank reasons on blurred half:", sorted({s['blank_reason'] for s in blurry}))
    print("INTEGRITY CHECK PASSED")


if __name__ == "__main__":
    main()
