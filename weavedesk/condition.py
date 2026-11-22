"""Read midribs, mounting seams and pale flecks using supported leaf pixels."""

from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .art import base_rgba, magnitude_overlay
from .model import InputError


LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)
PAINT = (248, 202, 102)
WORN = (244, 126, 105)
UNKNOWN = (165, 179, 192)


def stable_number(value, decimals=5):
    if not np.isfinite(value):
        return None
    return round(float(value), decimals)


def fraction(mask, denominator):
    n = int(denominator.sum())
    return round(100 * int((mask & denominator).sum()) / n, 2) if n else None


def cross_slices(frame, area, linear, settings, radius):
    points = np.asarray([settings['start'], settings['end']])
    w, h = frame.extent
    anchors = points * [w - 1, h - 1]
    vector = anchors[1] - anchors[0]
    length = np.linalg.norm(vector)
    if length < 9:
        raise InputError('Mark a path spanning at least nine registered pixels.')
    tangent = vector / length
    normal = np.asarray([-tangent[1], tangent[0]])
    steps = np.linspace(0., 1., settings['samples'])
    centers = anchors[0] + steps[:, None] * vector
    offsets = np.arange(-radius, radius + 1)
    samples = centers[:, None, :] + offsets[None, :, None] * normal
    x, y = samples[..., 0], samples[..., 1]
    coordinates = [y.ravel(), x.ravel()]
    valid = (area & frame.valid & ~frame.clipped).astype(np.float32)
    covered = ndimage.map_coordinates(valid, coordinates, order=1, mode='constant', cval=0).reshape(x.shape) > .999
    luma = ndimage.map_coordinates(linear @ LUMA, coordinates, order=1, mode='constant', cval=0).reshape(x.shape)
    if frame.scale:
        physical = np.linalg.norm(vector * np.asarray(frame.scale))
        side_scale = np.linalg.norm(normal * np.asarray(frame.scale))
    else:
        physical = length
        side_scale = 1.
    return centers, offsets, luma, covered, steps * physical, side_scale


def observed_runs(states):
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    return [slice(a, b) for a, b in zip(starts, np.r_[starts[1:], len(states)], strict=True)]


def annotated_runs(frame, centers, runs, states, palette):
    image = base_rgba(frame)
    draw = ImageDraw.Draw(image)
    for segment in runs:
        points = [tuple(item) for item in centers[segment]]
        color = palette[states[segment.start]]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
        else:
            draw.line(points, fill=(25, 31, 34), width=7)
            draw.line(points, fill=color, width=4)
    return image


def scan_marking(frame, area, linear, settings):
    radius = settings['radius']
    centers, offsets, luma, valid, distance, normal_scale = cross_slices(frame, area, linear, settings, radius)
    core = np.abs(offsets) <= max(1, radius // 3)
    reference = np.abs(offsets) > max(1, radius * .63)
    if not np.any(reference):
        raise InputError('Include supported blade tissue on both sides of the marked axis.')
    adequate = (np.sum(valid[:, core], axis=1) >= (core.sum() + 1) // 2) & np.all(valid[:, reference], axis=1)
    contrast = np.zeros(len(centers), dtype=float)
    counts = np.zeros(len(centers), dtype=int)
    for row in np.flatnonzero(adequate):
        base = np.median(luma[row, reference])
        contrast[row] = np.median(luma[row, core & valid[row]]) - base
        counts[row] = int((valid[row] & (luma[row] >= base + settings['contrast'])).sum())
    states = np.full(len(centers), 'unmeasured', dtype=object)
    states[adequate & (contrast < settings['contrast'])] = 'low contrast'
    states[adequate & (contrast >= settings['contrast'])] = 'visible printed rule'
    runs = observed_runs(states)
    rows = []
    for segment in runs:
        flag = states[segment.start]
        selected = int(segment.stop - segment.start)
        x, y = centers[(segment.start + segment.stop - 1)//2] / np.asarray(frame.extent, dtype=float)
        rows.append({'condition': flag, 'first_station': stable_number(distance[segment.start], 3),
                     'last_station': stable_number(distance[segment.stop-1], 3),
                     'stations': selected,
                     'midpoint_normalized': [round(float(x), 6), round(float(y), 6)],
                     'median_cross_width': (stable_number(np.median(counts[segment])*normal_scale, 3)
                                            if flag == 'visible printed rule' else None)})
    metrics = {'unit': 'mm' if frame.scale else 'px', 'stations': len(states),
               'observed_percent': round(100*float(adequate.mean()), 2),
               'midrib_percent_of_observed': fraction(states == 'visible printed rule', adequate),
               'runs': rows, 'note': 'A straight band compares the bright midrib with its neighboring blade tissue.'}
    return metrics, annotated_runs(frame, centers, runs, states,
                                  {'unmeasured': UNKNOWN, 'low contrast': WORN, 'visible printed rule': PAINT})


def survey_joint(frame, area, linear, settings):
    radius = settings['radius']
    centers, offsets, luma, support, distance, side_scale = cross_slices(frame, area, linear, settings, radius)
    left = offsets <= -max(1, radius // 3)
    right = offsets >= max(1, radius // 3)
    good = (support[:, left].sum(axis=1) >= ((left.sum()+1)//2)) & (support[:, right].sum(axis=1) >= ((right.sum()+1)//2))
    delta = np.full(len(centers), np.nan)
    bands = np.full((len(centers), 2), np.nan)
    for row in np.flatnonzero(good):
        lp, rp = left & support[row], right & support[row]
        bands[row] = np.median(luma[row, lp]), np.median(luma[row, rp])
        delta[row] = bands[row, 1] - bands[row, 0]
    traces = np.full(len(centers), 'unmeasured', dtype='<U14')
    traces[good] = 'close levels'
    traces[good & (np.abs(delta) >= settings['contrast'])] = 'visible step'
    lines = []
    for i, station in enumerate(distance):
        lines.append({'position': stable_number(station, 3), 'left': stable_number(bands[i,0]),
                      'right': stable_number(bands[i,1]), 'right_minus_left': stable_number(delta[i]),
                      'status': traces[i], 'normalized_position': [round(float(x), 6) for x in centers[i]/np.array(frame.extent)]})
    positions = delta[good]
    metrics = {'unit': 'mm' if frame.scale else 'px', 'side_offset': stable_number(radius*side_scale, 3),
               'measured_stations': int(good.sum()), 'unmeasured_stations': int((~good).sum()),
               'median_absolute_step': stable_number(np.median(abs(positions))) if len(positions) else None,
               'stations': lines, 'note': 'A difference in reflected light is not a height measurement.'}
    return metrics, annotated_runs(frame, centers, observed_runs(traces), traces,
                                  {'unmeasured': UNKNOWN, 'close levels': PAINT, 'visible step': WORN})


@dataclass(frozen=True)
class AggregateCandidate:
    index: int
    size: int
    point: tuple
    contrast: float
    width: float
    height: float

    def public(self, frame):
        weights = frame.scale or (1, 1)
        return {'rank': self.index, 'registered_pixels': self.size,
                'normalized_center': [round(self.point[0]/frame.extent[0], 6), round(self.point[1]/frame.extent[1], 6)],
                'contrast': round(self.contrast, 5), 'bounding_width': round(self.width*weights[0], 3),
                'bounding_height': round(self.height*weights[1], 3)}


def aggregate_candidates(frame, area, linear, settings):
    radius = settings['radius']
    if settings['maximum'] < settings['minimum']:
        raise InputError('Maximum cluster size must not be below the minimum.')
    supported = frame.valid & area & ~frame.clipped
    luma = linear @ LUMA
    weighted = ndimage.gaussian_filter(np.where(supported, luma, 0), radius)
    weights = ndimage.gaussian_filter(supported.astype(np.float32), radius)
    base = np.divide(weighted, weights, where=weights > .55, out=np.zeros_like(luma))
    defined = supported & (weights > .55)
    score = np.where(defined, np.maximum(luma-base, 0), 0)
    seen = score > settings['contrast']
    labels, count = ndimage.label(seen, structure=np.ones((3,3)))
    boxes = ndimage.find_objects(labels)
    groups = []
    for idx in range(1, count+1):
        box = boxes[idx-1]
        local = labels[box] == idx
        size = int(local.sum())
        if size < settings['minimum'] or size > settings['maximum']:
            continue
        coords = np.argwhere(local)
        peak = float(score[box][local].max())
        y,x = coords.mean(axis=0) + [box[0].start,box[1].start]
        groups.append((peak,size,(x,y),box[1].stop-box[1].start,box[0].stop-box[0].start))
    groups.sort(key=lambda row: (-row[1], -row[0], row[2]))
    rows = [AggregateCandidate(i,size,point,peak,width,height).public(frame)
            for i,(peak,size,point,width,height) in enumerate(groups[:120],1)]
    metrics = {'unit': 'mm' if frame.scale else 'px', 'candidate_clusters': len(groups),
               'observed_pixels': int(defined.sum()), 'bright_fraction_of_support': fraction(seen,defined),
               'candidates': rows, 'note': 'Bright spots may be pale tissue, residue or glare.'}
    return metrics, magnitude_overlay(frame, score, defined, max(settings['contrast']*3,.025), unknown=area & ~defined)
