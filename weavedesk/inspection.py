import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .art import annotated_regions, base_rgba, magnitude_overlay
from .color import corrected_linear, float_list, linear_lab, quantile_rows
from .defects import gaussian_supported
from .geometry import active_region, analysis_mask, region_domain, specimen_frame
from .model import InputError, enumeration, identity, integer, scalar
from .photos import luminance, uint8_image


def side_preview(store, tile, photo_id, layer):
    frame = specimen_frame(store.blobs, tile, photo_id)
    if layer == "raw":
        return base_rgba(frame)
    corrected, _ = corrected_linear(tile, frame, layer in ("corrected", "regions"))
    if layer == "regions":
        return annotated_regions(frame, tile["regions"], corrected)
    return base_rgba(frame, corrected)


def cross_edge_mismatch(first, second, common):
    gray_a = luminance(first.linear)
    gray_b = luminance(second.linear)
    filtered_a, smooth_a = gaussian_supported(gray_a, common, 2.0)
    filtered_b, smooth_b = gaussian_supported(gray_b, common, 2.0)
    gy_a, gx_a = np.gradient(filtered_a)
    gy_b, gx_b = np.gradient(filtered_b)
    norm_a = np.hypot(gx_a, gy_a)
    norm_b = np.hypot(gx_b, gy_b)
    field = np.abs(gx_a * gx_b + gy_a * gy_b)
    unit = np.maximum(norm_a * norm_b, 1e-10)
    orientation_delta = 1 - np.clip(field / unit, 0, 1)
    certain = common & (smooth_a > 0.95) & (smooth_b > 0.95) & (np.minimum(norm_a, norm_b) > 0.003)
    return orientation_delta, certain


def side_statistics(first, second, domain):
    inside = domain & first.valid
    matched = inside & second.valid
    overlap = int(np.count_nonzero(matched))
    active_first = int(np.count_nonzero(inside))
    edge_error, confident = cross_edge_mismatch(first, second, matched)
    left, right = first.photo["role"], second.photo["role"]
    diagnostics = {
        "roles": [left, right],
        "support_overlap_px": overlap,
        "left_supported_px": active_first,
        "right_supported_px": int(np.count_nonzero(domain & second.valid)),
        "edge_positions_compared": int(np.count_nonzero(confident)),
        "edge_direction_disagreement": quantile_rows(edge_error[confident], (0.50, 0.90)) if confident.any() else None,
    }
    return diagnostics, edge_error, confident


def viewing_pair(store, tile, payload):
    first_id = identity(payload.get("first"), "First registered capture")
    second_id = identity(payload.get("second"), "Second registered capture")
    if first_id == second_id:
        raise InputError("Select two different registered captures.")
    branch = enumeration(payload.get("mode", "checker"), ("checker", "split", "edges", "support"), "Viewing mode")
    first = specimen_frame(store.blobs, tile, first_id)
    second = specimen_frame(store.blobs, tile, second_id)
    region = active_region(tile, payload.get("region_id"))
    domain = region_domain(first.extent, tile, region)
    stats, field, confident = side_statistics(first, second, domain)
    if branch == "edges":
        picture = magnitude_overlay(first, field, confident, 1.0)
    elif branch == "support":
        picture = support_picture(first, second, domain)
    else:
        width, height = first.extent
        if branch == "checker":
            size = integer(payload.get("cells", 8), "Checkerboard cells", lower=3, upper=20)
            step = max(1, int(round(min(width, height) / size)))
            yy, xx = np.ogrid[:height, :width]
            switch = (xx // step + yy // step) % 2 == 0
        else:
            cut = scalar(payload.get("divider", 0.5), "Divider", lower=0.05, upper=0.95)
            divider = round(cut * width)
            switch = np.broadcast_to(np.arange(width)[None, :] < divider, (height, width))
        image = np.where(switch[:, :, None], first.display(), second.display())
        visible = np.where(switch, first.valid, second.valid)
        picture = uint8_image(image, visible.astype(np.float32))
        if branch == "split":
            draw = ImageDraw.Draw(picture)
            draw.line((divider, 0, divider, height), fill=(242, 221, 174), width=2)
    return picture, stats


def support_picture(first, second, area):
    both = first.valid & second.valid & area
    left_only = first.valid & ~second.valid & area
    right_only = ~first.valid & second.valid & area
    faded = first.display() * 0.30
    faded[both] = (0.36, 0.65, 0.58)
    faded[left_only] = (0.84, 0.51, 0.37)
    faded[right_only] = (0.47, 0.63, 0.84)
    return uint8_image(faded, np.ones(area.shape, dtype=np.float32))


def quality_cell(frame, support, detail, bounds):
    top, bottom, left, right = bounds
    selection = np.s_[top:bottom, left:right]
    pixels = support[selection]
    clip = frame.clipped[selection] & pixels
    count = int(np.count_nonzero(pixels))
    if count == 0:
        return None
    solid = detail[selection][pixels]
    width, height = frame.extent
    return {
        "bounds": [left / width, top / height, right / width, bottom / height],
        "supported": count,
        "supported_percent": round(100 * count / pixels.size, 2),
        "clipped_percent": round(100 * np.count_nonzero(clip) / count, 2),
        "detail_quantiles": quantile_rows(solid, (0.20, 0.50, 0.90), 6),
    }


def focus_diagnostics(frame, domain, grid):
    support = analysis_mask(frame, domain, minimum=36, allow_clipped=True)
    image = luminance(frame.linear)
    base, weight = gaussian_supported(image, support, 1.5)
    detail = np.abs(image - base)
    detail[weight < 0.9] = 0
    h, w = support.shape
    y_edges = np.rint(np.linspace(0, h, grid + 1)).astype(int)
    x_edges = np.rint(np.linspace(0, w, grid + 1)).astype(int)
    patches = []
    for top, bottom in zip(y_edges, y_edges[1:]):
        for left, right in zip(x_edges, x_edges[1:]):
            cell = quality_cell(frame, support, detail, (top, bottom, left, right))
            if cell:
                patches.append(cell)
    population = detail[support & (weight >= 0.9)]
    covered = support & ~frame.clipped
    raw = frame.linear[covered]
    ranges = float_list(np.quantile(linear_lab(raw)[:, 0], (0.1, 0.5, 0.9))) if len(raw) else None
    return {
        "cells": patches,
        "global_detail": quantile_rows(population, (0.20, 0.50, 0.90), 6) if len(population) else None,
        "unclipped_lightness": ranges,
        "patches_with_clipping": sum(item["clipped_percent"] > 2 for item in patches),
        "meaning": "Image detail represents recorded local contrast. It can also reflect the glaze texture.",
    }
