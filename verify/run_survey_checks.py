#!/usr/bin/env python3 -P
"""End-to-end smoke test for the survey direction and repeat spacing flow."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weavedesk.web import create_app  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok -", message)


def main():
    data_dir = Path(tempfile.mkdtemp(prefix="weavedesk-verify-"))
    try:
        app = create_app(data_dir)
        client = app.test_client()

        print("training data")
        res = client.post("/api/training", json={})
        check(res.status_code == 201, f"training loaded ({res.status_code})")
        tiles = res.get_json()["items"]
        tile_id = tiles[0]["id"]

        tile = client.get(f"/api/specimens/{tile_id}").get_json()
        base = next(p for p in tile["photos"] if p["registration_state"] == "base")
        second = next(p for p in tile["photos"] if p["registration_state"] == "registered")
        inspection_regions = [r for r in tile["regions"] if r["purpose"] != "ignore"]
        check(len(inspection_regions) >= 2, "specimen has multiple inspection regions")

        print("survey direction validation")
        res = client.post(f"/api/specimens/{tile_id}/survey/preview",
                          json={"survey_axis": 47})
        check(res.status_code == 422, "a diagonal direction is rejected by the cradle rule")

        res = client.post(f"/api/specimens/{tile_id}/survey/preview",
                          json={"survey_axis": 90, "pins": [[0.1, 0.5], [0.9, 0.5]]})
        check(res.status_code == 200, "90 degree preview renders")
        check(res.headers["X-Survey-Summary"], "summary header present")
        summary = json.loads(res.headers["X-Survey-Summary"])
        check(summary["survey_axis_deg"] == 90.0, "summary carries adopted axis")
        check(all("skew_deg" in row for row in summary["regions"]), "per-region skew reported")
        check(res.data[:8] == b"\x89PNG\r\n\x1a\n", "preview is a PNG")

        print("saving the plan")
        version = tile["version"]
        res = client.put(f"/api/specimens/{tile_id}/survey",
                         json={"version": version, "survey_axis": 90,
                               "pins": [[0.1, 0.5], [0.9, 0.5]]})
        check(res.status_code == 200, f"plan saved ({res.status_code} {res.data[:200]})")
        updated = res.get_json()
        check(updated["survey_plan"]["survey_axis_deg"] == 90.0, "plan stored on specimen")
        check(set(updated["survey_plan"]["region_ids"]) ==
              {r["id"] for r in inspection_regions}, "plan records the sample regions")

        print("repeat spacing check")
        res = client.post(f"/api/specimens/{tile_id}/results", json={
            "kind": "weave",
            "photo_id": base["id"],
            "comparison_id": second["id"],
            "region_id": inspection_regions[0]["id"],
            "options": {"analysis_mode": "repeat", "segment_length": 42, "band_width": 14},
        })
        check(res.status_code == 201, f"repeat result created ({res.status_code} {res.data[:300]})")
        result = res.get_json()
        metrics = result["metrics"]
        check(metrics["mode"] == "repeat", "result tagged as repeat reading")
        check(metrics["survey_axis_deg"] == 90.0, "repeat uses the adopted axis")
        check(len(metrics["segments"]) > 0, "segments were produced")
        check({s["status"] for s in metrics["segments"]} <= {"measured", "blank"},
              "every segment is measured or intentionally blank")
        blanks = [s for s in metrics["segments"] if s["status"] == "blank"]
        for segment in blanks:
            check(bool(segment["blank_reason"]), "blank segments carry a reason, never a borrowed peak")
            check(segment["pitch_a"] is None and segment["pitch_b"] is None,
                  "blank segments have no fabricated pitches")
        measured = [s for s in metrics["segments"] if s["status"] == "measured"]
        if measured:
            sample = measured[0]
            check(sample["pitch_a"] > 0 and sample["pitch_b"] > 0, "measured pitches are positive")
            check(abs(sample["pitch_change_percent"]) < 100, "pitch change is finite")
        check(result["pictures"]["overlay"] and result["pictures"]["paired"],
              "overlay and paired capture pictures retained")

        res = client.get(f"/api/results/{result['id']}/metrics.csv")
        check(res.status_code == 200 and b"blank_reason" in res.data, "repeat CSV export works")

        print("ordinary spectral weave reading unchanged")
        res = client.post(f"/api/specimens/{tile_id}/results", json={
            "kind": "weave",
            "photo_id": base["id"],
            "region_id": inspection_regions[0]["id"],
            "options": {"analysis_mode": "spectral", "reference_axis": 0, "tolerance": 24,
                        "minimum_pitch": 5, "maximum_pitch": 32},
        })
        check(res.status_code == 201, f"spectral result created ({res.status_code} {res.data[:300]})")
        spectral = res.get_json()["metrics"]
        check("rows" in spectral and len(spectral["rows"]) == 2, "two yarn family rows as before")
        check(spectral.get("mode") != "repeat", "spectral reading keeps its original shape")

        print("repeat requires the saved plan")
        res = client.delete(f"/api/specimens/{tile_id}/survey",
                            json={"version": updated["version"]})
        check(res.status_code == 200, "plan cleared")
        res = client.post(f"/api/specimens/{tile_id}/results", json={
            "kind": "weave",
            "photo_id": base["id"],
            "comparison_id": second["id"],
            "options": {"analysis_mode": "repeat", "segment_length": 42, "band_width": 14},
        })
        check(res.status_code == 422, "repeat check refused without an adopted direction")

        print("repeat parameter and input validation")
        fresh = client.get(f"/api/specimens/{tile_id}").get_json()
        client.put(f"/api/specimens/{tile_id}/survey",
                   json={"version": fresh["version"], "survey_axis": 0, "pins": []})
        res = client.post(f"/api/specimens/{tile_id}/results", json={
            "kind": "weave", "photo_id": base["id"], "comparison_id": base["id"],
            "options": {"analysis_mode": "repeat", "segment_length": 42, "band_width": 14},
        })
        check(res.status_code == 422, "identical earlier/later photos are rejected")
        res = client.post(f"/api/specimens/{tile_id}/results", json={
            "kind": "weave", "photo_id": base["id"], "comparison_id": second["id"],
            "options": {"analysis_mode": "repeat", "segment_length": 999, "band_width": 14},
        })
        check(res.status_code == 422, "out-of-range segment length is rejected")
        res = client.post(f"/api/specimens/{tile_id}/survey/preview",
                          json={"survey_axis": 0, "pins": [[0.5, 0.5], [0.5, 0.5]]})
        check(res.status_code == 422, "coincident registration pins are rejected")
        tile_now = client.get(f"/api/specimens/{tile_id}").get_json()
        check(tile_now.get("survey_plan") is not None, "plan restored for later runs")

        print("plan without registration pins")
        res = client.put(f"/api/specimens/{tile_id}/survey",
                         json={"version": tile_now["version"], "survey_axis": 90, "pins": []})
        check(res.status_code == 200, "pin-less plan accepted")
        pinless = res.get_json()["survey_plan"]
        check(pinless["pins"] == [] and pinless["pin_line_angle_deg"] is None,
              "pin-less plan stores no pins and no pin angle")
        res = client.post(f"/api/specimens/{tile_id}/results", json={
            "kind": "weave", "photo_id": base["id"], "comparison_id": second["id"],
            "options": {"analysis_mode": "repeat", "segment_length": 42, "band_width": 14},
        })
        check(res.status_code == 201, "repeat check works without pins")
        check(res.get_json()["metrics"]["survey_axis_deg"] == 90.0, "adopted axis still applied")

        print("ALL CHECKS PASSED")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
