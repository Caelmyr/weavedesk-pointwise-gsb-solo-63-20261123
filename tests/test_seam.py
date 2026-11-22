"""Tests for the broken-ribbon seam workbench.

The synthetic fragment pair is generated with a known ground-truth notch,
blue stripe groups, and frayed warp ends.  These tests assert that the engine
picks the middle notch from visible evidence, pairs only the warp ends that
can genuinely join, leaves the unphotographed gap empty, and tracks every
line-scan color back to a real source pixel.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = "/opt/pointwise-env/site-packages"
for path in (str(ROOT), SITE):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
from PIL import Image

from weavedesk.model import InputError
from weavedesk.seam import (
    GAP_PADDING_PX,
    blue_stripes,
    composite_band,
    evaluate_notches,
    line_scan,
    warp_ends,
)
from weavedesk.seamdemo import render_demo_pair
from weavedesk.seamservice import add_line_scan, create_seam, demo_photo_records, demo_specimen, load_demo_seam
from weavedesk.store import Store


class SeamEngineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.temp.name) / "data"))
        pair = render_demo_pair()
        self.pair = pair
        records = demo_photo_records(self.store, pair)
        self.specimen = demo_specimen(self.store, records)
        self.records = records
        self.fragments = {}
        from weavedesk.seamservice import _load_fragment

        for side in ("left", "right"):
            spec = records[side]
            self.fragments[side], _ = _load_fragment(
                self.store, side,
                {
                    "photo_id": spec["photo"]["id"],
                    "corners": spec["corners"],
                    "outline": spec["outline"],
                    "pixels_per_mm": spec["pixels_per_mm"],
                },
            )

    def tearDown(self):
        self.temp.cleanup()

    def test_blue_stripes_align_on_both_fragments(self):
        left_stripes, _ = blue_stripes(self.fragments["left"])
        right_stripes, _ = blue_stripes(self.fragments["right"])
        self.assertEqual(len(left_stripes), 2)
        self.assertEqual(len(right_stripes), 2)
        for left, right in zip(left_stripes, right_stripes):
            self.assertAlmostEqual(left["y"], right["y"], delta=2.0)

    def test_frayed_ends_match_ground_truth_pairs(self):
        left_threads = warp_ends(self.fragments["left"])
        right_threads = warp_ends(self.fragments["right"])
        left_ys = sorted(item["y"] for item in left_threads)
        right_ys = sorted(item["y"] for item in right_threads)
        # Two pair-able rows (44, 108) plus one lone end on each side.
        self.assertEqual(len(left_threads), 3)
        self.assertEqual(len(right_threads), 3)
        for ground_truth in (44, 78, 108):
            self.assertTrue(any(abs(y - ground_truth) <= 3 for y in left_ys), left_ys)
        for ground_truth in (44, 108, 132):
            self.assertTrue(any(abs(y - ground_truth) <= 3 for y in right_ys), right_ys)

    def test_middle_notch_wins_from_visible_evidence(self):
        chosen, report, _ = evaluate_notches(
            self.fragments["left"], self.fragments["right"], self.pair["notch_step_px"]
        )
        self.assertEqual(chosen["notch"], self.pair["ground_truth_notch"])
        scores = {item["notch"]: item["score"] for item in report}
        self.assertLess(scores[0], scores[-1])
        self.assertLess(scores[0], scores[1])
        # The two genuine warp pairs join only at the middle notch.
        self.assertEqual(chosen["matched_thread_count"], 2)
        # The lone ends remain explicitly unmatched.
        self.assertTrue(chosen["unmatched_left_threads"])
        self.assertTrue(chosen["unmatched_right_threads"])

    def test_composite_never_fills_the_gap_or_blends_pixels(self):
        chosen, _, _ = evaluate_notches(
            self.fragments["left"], self.fragments["right"], self.pair["notch_step_px"]
        )
        canvas, ownership, supported, layout = composite_band(
            self.fragments["left"], self.fragments["right"], chosen["shift_px"]
        )
        # The central seam columns are unsupported: no invented pixels.
        seam_column = layout["seam_x"]
        gap_columns = slice(seam_column - GAP_PADDING_PX + 1, seam_column + GAP_PADDING_PX - 1)
        self.assertFalse(supported[:, gap_columns].any())
        # Every supported pixel belongs to exactly one named fragment.
        self.assertTrue(set(np.unique(ownership[supported])).issubset({"left", "right"}))
        # Pixels copied from a fragment equal its original rectified values.
        left = self.fragments["left"]
        offset_x, offset_y = layout["left_offset"]
        probe_y, probe_x = 20, 100
        self.assertTrue(supported[probe_y + offset_y, probe_x + offset_x])
        np.testing.assert_allclose(
            canvas[probe_y + offset_y, probe_x + offset_x],
            left.linear[probe_y, probe_x],
            rtol=1e-6,
        )

    def test_line_scan_blanks_unseen_stations_and_keeps_provenance(self):
        chosen, _, _ = evaluate_notches(
            self.fragments["left"], self.fragments["right"], self.pair["notch_step_px"]
        )
        shift = chosen["shift_px"]
        canvas, ownership, supported, layout = composite_band(
            self.fragments["left"], self.fragments["right"], shift
        )
        scan = line_scan(
            self.fragments["left"], self.fragments["right"], layout, ownership, supported,
            [0.05, 0.5], [0.95, 0.5], 120, 3,
        )
        blanks = [row for row in scan["rows"] if row["fragment"] is None]
        measured = [row for row in scan["rows"] if row["fragment"] is not None]
        self.assertTrue(blanks, "the unphotographed gap must produce blank stations")
        for row in blanks:
            self.assertIsNone(row["lab"])
            self.assertIsNone(row["source_pixel"])
            self.assertIsNone(row["source_photo_id"])
        for row in measured:
            self.assertIsNotNone(row["lab"])
            sx, sy = row["source_pixel"]
            photo = next(item for item in self.specimen["photos"] if item["id"] == row["source_photo_id"])
            self.assertTrue(0 <= sx < photo["width"] and 0 <= sy < photo["height"])

    def test_line_scan_requires_a_visible_crossing(self):
        chosen, _, _ = evaluate_notches(
            self.fragments["left"], self.fragments["right"], self.pair["notch_step_px"]
        )
        _, _, supported, layout = composite_band(
            self.fragments["left"], self.fragments["right"], chosen["shift_px"]
        )
        ownership = np.full(supported.shape, "", dtype=object)
        with self.assertRaises(InputError):
            line_scan(
                self.fragments["left"], self.fragments["right"], layout, ownership, supported,
                [0.46, 0.0], [0.47, 1.0], 40, 1,
            )


class SeamServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.temp.name) / "data"))

    def tearDown(self):
        self.temp.cleanup()

    def test_demo_seam_persists_with_composite_and_report(self):
        record, pair = load_demo_seam(self.store)
        self.assertEqual(record["chosen_notch"], 0)
        self.assertTrue(record["composite_digest"])
        self.assertEqual(len(record["notch_report"]), 3)
        # The stored composite PNG decodes to a real image.
        from weavedesk.art import load_artifact_picture

        image = load_artifact_picture(self.store.blobs, record["composite_digest"])
        self.assertIsInstance(image, Image.Image)
        self.assertGreater(image.width, 0)

    def test_notch_preview_renders_each_legal_placement(self):
        record, _ = load_demo_seam(self.store)
        from weavedesk.seamservice import preview_notch

        widths = set()
        for notch in (-1, 0, 1):
            image, chosen = preview_notch(self.store, record["id"], {"notch": notch})
            self.assertEqual(chosen["notch"], notch)
            self.assertEqual(image.mode, "RGBA")
            self.assertGreater(image.width, 0)
            self.assertGreaterEqual(image.height, record["layout"]["height"])
            widths.add(image.width)
        # Shifting vertically changes height, never the band length.
        self.assertEqual(len(widths), 1)

    def test_add_line_scan_persists_blank_and_measured_stations(self):
        record, _ = load_demo_seam(self.store)
        updated, scan = add_line_scan(
            self.store, record["id"],
            {"start": [0.05, 0.5], "end": [0.95, 0.5], "stations": 90, "band_pixels": 3},
        )
        self.assertEqual(len(updated["line_scans"]), 1)
        self.assertGreater(scan["blank_stations"], 0)
        self.assertGreater(scan["supported_stations"], 0)

    def test_create_seam_rejects_unknown_photo(self):
        with self.assertRaises(InputError):
            create_seam(self.store, {
                "title": "bad",
                "left": {"photo_id": "0" * 24, "corners": self._corners(), "outline": self._outline()},
                "right": {"photo_id": "0" * 24, "corners": self._corners(), "outline": self._outline()},
            })

    @staticmethod
    def _corners():
        return [[0.1, 0.3], [0.9, 0.3], [0.9, 0.7], [0.1, 0.7]]

    @staticmethod
    def _outline():
        return [
            [0.1, 0.3], [0.4, 0.3], [0.42, 0.4], [0.41, 0.5],
            [0.43, 0.6], [0.4, 0.7], [0.1, 0.7],
        ]


if __name__ == "__main__":
    unittest.main(verbosity=2)
