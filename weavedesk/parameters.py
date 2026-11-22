from .model import dictionary, enumeration, flag, identity, integer, numeric_pair, scalar


TOOLS = {
    "weave": {
        "label": "Yarn spacing and angle",
        "description": "Estimate the two main yarn directions and spacing from a visible fabric window.",
        "fields": {
            "analysis_mode": {"kind": "choice", "label": "Reading mode", "default": "spectral", "choices": ["spectral", "repeat"]},
            "reference_axis": {"kind": "number", "label": "Approximate first yarn direction (degrees)", "default": 0, "min": 0, "max": 179, "step": 1},
            "tolerance": {"kind": "number", "label": "Search half-angle (degrees)", "default": 24, "min": 6, "max": 35, "step": 1},
            "minimum_pitch": {"kind": "number", "label": "Minimum spacing (px)", "default": 5, "min": 3, "max": 25, "step": 1},
            "maximum_pitch": {"kind": "number", "label": "Maximum spacing (px)", "default": 32, "min": 8, "max": 52, "step": 1},
            "segment_length": {"kind": "number", "label": "Repeat segment length (px)", "default": 42, "min": 12, "max": 260, "step": 1},
            "band_width": {"kind": "number", "label": "Repeat profile band width (px)", "default": 14, "min": 3, "max": 120, "step": 1},
        },
    },
    "dots": {
        "label": "Yarn knot census",
        "description": "Measure isolated dark yarn knots, their apparent footprint and elongation on the registered sample.",
        "fields": {
            "radius": {"kind": "integer", "label": "Paper background radius (px)", "default": 8, "min": 3, "max": 28},
            "contrast": {"kind": "number", "label": "Minimum knot contrast", "default": .13, "min": .03, "max": .65, "step": .01},
            "minimum_area": {"kind": "integer", "label": "Minimum footprint (px²)", "default": 5, "min": 2, "max": 80},
            "maximum_area": {"kind": "integer", "label": "Maximum footprint (px²)", "default": 144, "min": 24, "max": 1600},
        },
    },
    "color": {
        "label": "Color variation",
        "description": "Map the distance from the median specimen color and compare local cells.",
        "fields": {"grid": {"kind": "integer", "label": "Grid across", "default": 5, "min": 2, "max": 10}},
    },
    "coverage": {
        "label": "Fabric coverage",
        "description": "Estimate exposed mount, mixed material and fabric coverage from two marked sample patches.",
        "fields": {
            "bare_region": {"kind": "region", "purpose": "bare", "label": "Bare mounting cloth patch", "default": None},
            "glazed_region": {"kind": "region", "purpose": "glazed", "label": "Bound textile patch", "default": None},
            "ambiguity": {"kind": "number", "label": "Mixed margin", "default": 0.16, "min": 0.03, "max": 0.4, "step": 0.01},
            "max_residual_fraction": {"kind": "number", "label": "Off-axis tolerance", "default": 0.30, "min": 0.10, "max": 0.65, "step": 0.05},
            "minimum_island": {"kind": "integer", "label": "Minimum mount island px", "default": 22, "min": 4, "max": 1000},
        },
    },
    "defects": {
        "label": "Surface observations",
        "description": "Find local spots and elongated marks, then review observations by location.",
        "fields": {
            "sensitivity": {"kind": "number", "label": "Sensitivity", "default": 0.63, "min": 0, "max": 1, "step": 0.01},
            "spot_radius": {"kind": "number", "label": "Search radius px", "default": 5, "min": 1.2, "max": 32, "step": 0.1},
            "polarity": {"kind": "choice", "label": "Spot polarity", "default": "dark", "choices": ["dark", "light", "both"]},
            "minimum_area": {"kind": "integer", "label": "Minimum spot px", "default": 6, "min": 2, "max": 800},
            "maximum_area": {"kind": "integer", "label": "Maximum spot px", "default": 330, "min": 5, "max": 4500},
        },
    },
    "glare": {
        "label": "Polarized glare",
        "description": "Inspect the local intensity gained in parallel polarization on common visible pixels.",
        "capture_pair": ["parallel", "cross"],
        "fields": {
            "balance": {"kind": "choice", "label": "Exposure balance", "default": "auto", "choices": ["auto", "fixed"]},
            "fixed_gain": {"kind": "number", "label": "Fixed gain", "default": 1, "min": 0.2, "max": 5, "step": 0.05},
            "glare_fraction": {"kind": "number", "label": "Hotspot fraction", "default": 0.2, "min": 0.04, "max": 0.9, "step": 0.02},
            "minimum_hotspot": {"kind": "integer", "label": "Minimum hotspot px", "default": 30, "min": 3, "max": 5000},
            "balance_region": {"kind": "region", "label": "Exposure tie area", "default": None},
        },
    },
    "direction": {
        "label": "Raking light",
        "description": "Find fine marks that change sign when the light moves to the opposite side.",
        "capture_pair": ["rake-left", "rake-right"],
        "fields": {
            "balance": {"kind": "choice", "label": "Exposure balance", "default": "auto", "choices": ["auto", "fixed"]},
            "fixed_gain": {"kind": "number", "label": "Fixed gain", "default": 1, "min": 0.2, "max": 5, "step": 0.05},
            "scale_px": {"kind": "number", "label": "Shading radius px", "default": 8, "min": 2, "max": 30, "step": 0.5},
            "remove_broad": {"kind": "flag", "label": "Remove broad light tilt", "default": True},
            "strength_floor": {"kind": "number", "label": "Minimum contrast", "default": 0.04, "min": 0.01, "max": 0.3, "step": 0.01},
        },
    },
    "texture": {
        "label": "Surface texture",
        "description": "See local roughness and the direction and spacing of recurring texture.",
        "fields": {
            "background_radius": {"kind": "number", "label": "Background radius px", "default": 8, "min": 2, "max": 28, "step": 0.5},
            "edge_smooth": {"kind": "number", "label": "Edge smoothing px", "default": 1.2, "min": 0.5, "max": 5, "step": 0.1},
            "maximum_lag": {"kind": "integer", "label": "Maximum spacing px", "default": 20, "min": 4, "max": 48},
        },
    },
    "craze": {
        "label": "Fine-line network",
        "description": "Trace dark fissures and outline enclosed surface cells.",
        "fields": {
            "radius": {"kind": "number", "label": "Line width radius px", "default": 1.4, "min": 0.6, "max": 5, "step": 0.1},
            "sensitivity": {"kind": "number", "label": "Line sensitivity", "default": 0.72, "min": 0, "max": 1, "step": 0.02},
            "join_px": {"kind": "integer", "label": "Bridge width px", "default": 1, "min": 0, "max": 4},
            "component_minimum": {"kind": "integer", "label": "Minimum trace px", "default": 12, "min": 2, "max": 150},
            "minimum_cell": {"kind": "integer", "label": "Minimum closed-cell px", "default": 24, "min": 5, "max": 1200},
        },
    },
    "rim": {
        "label": "Survey margin color",
        "description": "Compare marked surface edges and depth bands with the interior color.",
        "fields": {
            "band": {"kind": "number", "label": "Band depth", "default": 8, "min": 0.5, "max": 70, "step": 0.5},
            "unit": {"kind": "choice", "label": "Band unit", "default": "percent", "choices": ["percent", "mm"]},
            "interior_multiple": {"kind": "number", "label": "Interior inset band multiples", "default": 2, "min": 1.1, "max": 3.5, "step": 0.1},
            "depth_bins": {"kind": "integer", "label": "Depth slices", "default": 4, "min": 2, "max": 9},
            "highlight_delta": {"kind": "number", "label": "Highlight from interior ΔE", "default": 13, "min": 3, "max": 45, "step": 1},
        },
    },
    "profile": {
        "label": "Line scan",
        "description": "Draw across a stripe or a surface boundary to inspect a line of color values.",
        "fields": {
            "start": {"kind": "point", "label": "Start", "default": [0.14, 0.37]},
            "end": {"kind": "point", "label": "End", "default": [0.86, 0.64]},
            "samples": {"kind": "integer", "label": "Stations", "default": 105, "min": 12, "max": 360},
            "band_pixels": {"kind": "number", "label": "Band width px", "default": 5, "min": 0, "max": 32, "step": 1},
        },
    },
    "blade": {
        "label": "Binding width",
        "description": "Measure supported binding strip cross-sections along two marked points.",
        "fields": {
            "start": {"kind": "point", "label": "Base end", "default": [.20, .733]},
            "end": {"kind": "point", "label": "Tip end", "default": [.80, .733]},
            "stations": {"kind": "integer", "label": "Cross-sections", "default": 24, "min": 5, "max": 150},
            "half_band": {"kind": "number", "label": "Cross-section reach", "default": .30, "min": .05, "max": .45, "step": .01},
            "bare_region": {"kind": "region", "purpose": "bare", "label": "Bare mounting cloth sample", "default": None},
            "glazed_region": {"kind": "region", "purpose": "glazed", "label": "Bound textile sample", "default": None},
        },
    },
    "reference": {
        "label": "Reference match",
        "description": "Compare this specimen with one frozen approved visual sample.",
        "fields": {
            "reference_id": {"kind": "reference", "label": "Approved visual reference", "default": None},
        },
    },
}


PATH_FIELDS = {
    "start": {"kind": "point", "label": "Start", "default": [.13, .48]},
    "end": {"kind": "point", "label": "End", "default": [.86, .48]},
    "samples": {"kind": "integer", "label": "Stations", "default": 82, "min": 10, "max": 210},
    "radius": {"kind": "integer", "label": "Sample half width px", "default": 16, "min": 4, "max": 54},
    "contrast": {"kind": "number", "label": "Signal threshold", "default": .055, "min": .005, "max": .6, "step": .005},
}
TOOLS.update({
    "marking": {"label": "Edge stitch visibility", "description": "Read light stitch sections and intervening lost support along a marked strip.", "fields": PATH_FIELDS},
    "joint": {"label": "Mount seam scan", "description": "Read both sides of one mounting or repair seam separately.", "fields": PATH_FIELDS},
    "aggregate": {"label": "Pale fiber flecks", "description": "Locate small clusters of brightness relative to the surrounding textile.", "fields": {
        "radius": {"kind": "integer", "label": "Background radius px", "default": 6, "min": 2, "max": 22},
        "contrast": {"kind": "number", "label": "Signal threshold", "default": .018, "min": .003, "max": .35, "step": .001},
        "minimum": {"kind": "integer", "label": "Minimum cluster px", "default": 4, "min": 1, "max": 800},
        "maximum": {"kind": "integer", "label": "Maximum cluster px", "default": 155, "min": 3, "max": 3500},
    }},
})


def parsed_option(key, field, raw):
    label = field["label"]
    kind = field["kind"]
    if kind == "integer":
        return integer(raw, label, lower=field["min"], upper=field["max"])
    if kind == "number":
        return scalar(raw, label, lower=field["min"], upper=field["max"])
    if kind == "choice":
        return enumeration(raw, field["choices"], label)
    if kind == "flag":
        return flag(raw, label)
    if kind == "point":
        return numeric_pair(raw, label, lower=0, upper=1)
    if kind == "region":
        return identity(raw, label) if raw else None
    if kind == "reference":
        return identity(raw, label)
    raise RuntimeError(f"Unknown measurement option {key}.")


def analysis_parameters(payload):
    raw = dictionary(payload, "Measurement")
    kind = enumeration(raw.get("kind"), TOOLS, "Measurement mode")
    raw_options = dictionary(raw.get("options", {}), "Measurement options")
    config = TOOLS[kind]
    options = {
        key: parsed_option(key, field, raw_options.get(key, field["default"]))
        for key, field in config["fields"].items()
    }
    return {
        "kind": kind,
        "options": options,
        "region_id": identity(raw["region_id"], "Sampling region") if raw.get("region_id") else None,
        "photo_id": identity(raw["photo_id"], "Photo ID") if raw.get("photo_id") else None,
        "comparison_id": identity(raw["comparison_id"], "Paired photo ID") if raw.get("comparison_id") else None,
        "apply_neutral": flag(raw.get("apply_neutral", True), "Apply card correction"),
    }
