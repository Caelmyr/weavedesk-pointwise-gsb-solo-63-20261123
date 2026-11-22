import math

import numpy as np
from PIL import Image
from scipy import ndimage

from .color import float_list, quantile_rows
from .defects import gaussian_supported, robust_noise
from .geometry import normalized_bbox
from .model import InputError
from .photos import luminance


NEIGHBORS = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)
LINE_CONNECTIVITY = ndimage.generate_binary_structure(2, 2)
PIXEL_EDGES = ((0, 1), (1, 0), (1, 1), (1, -1))
NOISE_MIN = 0.007
PIXEL_LIMIT = 3_100_000


def shifted_plane(padded, dy, dx, shape):
    height, width = shape
    return padded[1 + dy:height + 1 + dy, 1 + dx:width + 1 + dx]


def neighborhood(mask):
    extended = np.pad(mask, 1)
    return [shifted_plane(extended, dy, dx, mask.shape) for dy, dx in NEIGHBORS]


def transition_counts(neighbors):
    entered = np.zeros(neighbors[0].shape, dtype=np.uint8)
    for current, following in zip(neighbors, [*neighbors[1:], neighbors[0]]):
        entered += ~current & following
    return entered


def skeleton_step(active, primary):
    N, NE, E, SE, S, SW, W, NW = neighbors = neighborhood(active)
    degree = sum(neighbor.astype(np.uint8) for neighbor in (N, NE, E, SE, S, SW, W, NW))
    candidates = active & (degree >= 2) & (degree <= 6) & (transition_counts(neighbors) == 1)
    if primary:
        candidates &= ~(N & E & S) & ~(E & S & W)
    else:
        candidates &= ~(N & E & W) & ~(N & S & W)
    return active & ~candidates, int(np.count_nonzero(candidates))


def reduce_to_centerlines(mask, limit=28):
    result = mask.copy()
    exhausted = True
    replay = 0
    for iteration in range(1, limit + 1):
        result, one = skeleton_step(result, True)
        result, two = skeleton_step(result, False)
        replay = iteration
        if one + two == 0:
            exhausted = False
            break
    return result, replay, exhausted


def gradient_field(source, mask, radius):
    smooth, smooth_support = gaussian_supported(source, mask, radius)
    hxx = ndimage.gaussian_filter(smooth, 0.8, order=(0, 2)) * radius
    hyy = ndimage.gaussian_filter(smooth, 0.8, order=(2, 0)) * radius
    hxy = ndimage.gaussian_filter(smooth, 0.8, order=(1, 1)) * radius
    axis_sum = (hxx + hyy) / 2
    difference = (hxx - hyy) / 2
    radius_axis = np.hypot(difference, hxy)
    high = axis_sum + radius_axis
    low = axis_sum - radius_axis
    regularity = np.maximum(np.abs(high), 0.0001)
    thinness = np.clip(1 - np.abs(low) / regularity, 0, 1)
    validity = mask & (smooth_support > 0.93)
    return np.maximum(high, 0) * thinness, validity


def contrast_suppress(source, mask, radius):
    low, support = gaussian_supported(source, mask, radius * 4)
    contrast = np.maximum(low - source, 0)
    return contrast, support > 0.88


def retained_regions(binary, min_pixels):
    labels, count = ndimage.label(binary, LINE_CONNECTIVITY)
    if count == 0:
        return binary
    histogram = np.bincount(labels.ravel(), minlength=count + 1)
    approved = histogram >= min_pixels
    approved[0] = False
    return approved[labels]


def response_limits(response, contrast, support, sensitivity):
    population = response[support]
    signal = contrast[support]
    noise = robust_noise(population)
    wide_noise = robust_noise(signal)
    cutoff = max(NOISE_MIN, noise * (3.2 - 2.1 * sensitivity))
    tone = max(0.012, wide_noise * (2.1 - 1.1 * sensitivity))
    return cutoff, tone, (noise, wide_noise)


def threshold_centerline(first, fine, coarse, support, controls):
    threshold, tone, noise = response_limits(first, fine, support, controls["sensitivity"])
    binary = support & (first >= threshold) & (fine >= tone)
    if controls["join_px"] > 0:
        close_radius = controls["join_px"]
        structure = ndimage.iterate_structure(ndimage.generate_binary_structure(2, 1), close_radius)
        binary = ndimage.binary_closing(binary, structure=structure)
        binary &= support & (coarse > tone * 0.58)
    area = retained_regions(binary, controls["component_minimum"])
    centerlines, passes, unsettled = reduce_to_centerlines(area)
    return centerlines, area, {
        "threshold": round(threshold, 6),
        "contrast_floor": round(tone, 6),
        "noise": float_list(noise, 7),
        "thinning_passes": passes,
        "thinning_reached_limit": unsettled,
    }


def connected_adjacencies(mask, dy, dx):
    extended = np.pad(mask, 1)
    other = shifted_plane(extended, dy, dx, mask.shape)
    link = mask & other
    if dy and dx:
        adjacent_y = shifted_plane(extended, dy, 0, mask.shape)
        adjacent_x = shifted_plane(extended, 0, dx, mask.shape)
        link &= ~adjacent_y & ~adjacent_x
    return link


def degree_map(mask):
    degree = np.zeros(mask.shape, dtype=np.uint8)
    for dy, dx in NEIGHBORS:
        degree += connected_adjacencies(mask, dy, dx)
    return degree


def effective_edge_length(mask, scale):
    scale_x, scale_y = scale or (1, 1)
    total = 0.0
    count = 0
    for dy, dx in PIXEL_EDGES:
        edges = np.count_nonzero(connected_adjacencies(mask, dy, dx))
        count += int(edges)
        total += edges * math.hypot(dy * scale_y, dx * scale_x)
    return total, count


def branch_nodes(degree, network):
    candidates = ((degree >= 3) | (degree == 1)) & network
    labels, count = ndimage.label(candidates, LINE_CONNECTIVITY)
    if count == 0:
        return [], labels
    nodes = []
    height, width = network.shape
    for index, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        rr, cc = np.nonzero(labels[region] == index)
        val = degree[region][rr, cc]
        interior = val >= 3
        if interior.any():
            primitive = "junction"
            rr, cc = rr[interior], cc[interior]
        else:
            primitive = "endpoint"
        v = (float(np.mean(rr)) + region[0].start + 0.5) / height
        u = (float(np.mean(cc)) + region[1].start + 0.5) / width
        nodes.append({"type": primitive, "center": [round(u, 6), round(v, 6)], "pixels": len(rr)})
    return nodes, labels


def closed_mesh(mask, area, frame, minimum):
    filled = ndimage.binary_fill_holes(mask, structure=ndimage.generate_binary_structure(2, 1))
    interiors = filled & ~mask
    labels, count = ndimage.label(interiors)
    if not count:
        return []
    regions = []
    for index, bounds in enumerate(ndimage.find_objects(labels), 1):
        if bounds is None:
            continue
        selected = labels[bounds] == index
        size = int(np.count_nonzero(selected))
        if size < minimum or not np.all(area[bounds][selected]):
            continue
        regions.append({
            "pixels": size,
            "bounds": float_list(normalized_bbox(bounds, frame.extent), 5),
            "mm2": round(size * frame.scale[0] * frame.scale[1], 4) if frame.scale else None,
        })
    return sorted(regions, key=lambda item: -item["pixels"])


def craze_overlay(frame, network, footprint, nodes):
    rgb = frame.display() * 0.59
    rgb[footprint] = rgb[footprint] * 0.68 + np.array([0.42, 0.37, 0.29], dtype=np.float32)
    enhanced = ndimage.maximum_filter(network, size=3)
    rgb[enhanced] = (0.91, 0.74, 0.43)
    width, height = frame.extent
    if nodes:
        for item in nodes[:350]:
            x = min(int(item["center"][0] * width), width - 1)
            y = min(int(item["center"][1] * height), height - 1)
            dot = np.s_[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            rgb[dot] = (0.53, 0.91, 0.82) if item["type"] == "junction" else (0.98, 0.66, 0.62)
    values = np.rint(np.clip(rgb * 255, 0, 255)).astype(np.uint8)
    alpha = np.where(frame.valid, 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack((values, alpha)))


def measure_craze(frame, linear, area, options):
    source = luminance(linear)
    candidate = area & frame.valid & ~frame.clipped
    if source.size > PIXEL_LIMIT:
        raise InputError("Crackle tracing supports registered planes up to 3.1 megapixels.")
    filtered, valid = gradient_field(source, candidate, options["radius"])
    fine, support = contrast_suppress(source, candidate, options["radius"])
    coarse, margin = contrast_suppress(source, candidate, 2 * options["radius"])
    support &= valid & margin
    if np.count_nonzero(support) < 100:
        raise InputError("Choose a larger area with contiguous, unclipped image support.")
    thin, binary, thresholds = threshold_centerline(filtered, fine, coarse, support, options)
    degrees = degree_map(thin)
    length, edges = effective_edge_length(thin, frame.scale)
    nodes, _ = branch_nodes(degrees, thin)
    cells = closed_mesh(thin, support, frame, options["minimum_cell"])
    supported = int(np.count_nonzero(support))
    if frame.scale:
        area_mm = supported * frame.scale[0] * frame.scale[1]
        density = length / area_mm
    else:
        density = length / supported
    metrics = {
        "network_length": round(length, 5),
        "length_unit": "mm" if frame.scale else "px",
        "density": round(float(density), 6),
        "edge_links": edges,
        "traced_pixels": int(np.count_nonzero(thin)),
        "junctions": sum(item["type"] == "junction" for item in nodes),
        "endpoints": sum(item["type"] == "endpoint" for item in nodes),
        "closed_cell_count": len(cells),
        "largest_cells": cells[:45],
        "cell_size_quantiles_px": quantile_rows([item["pixels"] for item in cells]) if cells else None,
        "support_pixels": supported,
        "thresholds": thresholds,
        "note": "An image trace of thin dark lines. Verify finish marks and scratches against the source photo.",
    }
    return metrics, craze_overlay(frame, thin, binary, nodes)
