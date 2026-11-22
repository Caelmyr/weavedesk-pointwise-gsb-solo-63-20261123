#!/usr/bin/env python3
"""HTTP-level smoke test for the relief and cover-edge API flow."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weavedesk.web import create_app
from weavedesk.store import Store, make_id
from weavedesk.geometry import DEFAULT_CORNERS, canvas_dimensions
from weavedesk.photos import png_bytes, store_capture
from verify_relief import build_tile, box_outline, CLUSTERS  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp))
        with store.transaction(write=True) as conn:
            tile = store._insert_tile(conn, build_tile(store), "HTTP tile")
        app = create_app(tmp)
        client = app.test_client()

        # Configuration exposes the relief tool.
        config = client.get("/api/configuration").get_json()
        assert "relief" in config["tools"], "relief tool missing from configuration"
        assert config["tools"]["relief"]["capture_pair"] == ["rake-left", "rake-right"]
        print("[PASS] configuration exposes pile relief tool with raking pair")

        tile_id = tile["id"]
        left_id = next(p["id"] for p in tile["photos"] if p["role"] == "rake-left")
        right_id = next(p["id"] for p in tile["photos"] if p["role"] == "rake-right")
        payload = {
            "kind": "relief",
            "photo_id": left_id,
            "comparison_id": right_id,
            "region_id": None,
            "options": {
                "shading_radius": 7.0, "sensitivity": 0.62,
                "match_tolerance_px": 60.0, "cross_axis_px": 10.0,
                "min_shadow_pixels": 30, "calibration_anchor": None,
            },
            "anchors": [
                {"label": "Edge pile", "pile_height_mm": CLUSTERS[0]["height"],
                 "vertices": box_outline(CLUSTERS[0]["cx"], CLUSTERS[0]["cy"], CLUSTERS[0]["half"])},
                {"label": "Tall pile", "pile_height_mm": CLUSTERS[1]["height"],
                 "vertices": box_outline(CLUSTERS[1]["cx"], CLUSTERS[1]["cy"], CLUSTERS[1]["half"])},
            ],
        }
        response = client.post(f"/api/specimens/{tile_id}/results", json=payload)
        assert response.status_code == 201, response.get_data(as_text=True)
        result = response.get_json()
        assert result["kind"] == "relief"
        assert result["metrics"]["matched_feature_count"] == 3
        assert result["pictures"]["overlay"]
        print("[PASS] relief measurement created via HTTP with 3 matched clusters")

        # Overlay image is served.
        image_response = client.get(f"/api/results/{result['id']}/image/overlay")
        assert image_response.status_code == 200
        assert image_response.mimetype == "image/png"
        Image.open(__import__("io").BytesIO(image_response.data)).verify()
        print("[PASS] relief overlay PNG served")

        # CSV export works for the new kind.
        csv_response = client.get(f"/api/results/{result['id']}/metrics.csv")
        assert csv_response.status_code == 200
        text = csv_response.data.decode("utf-8-sig")
        assert "estimated_height_mm" in text and "R00" in text
        print("[PASS] relief metrics CSV export contains feature rows")

        # Missing anchors are rejected with 422.
        bad = json.loads(json.dumps(payload))
        bad["anchors"] = payload["anchors"][:1]
        rejected = client.post(f"/api/specimens/{tile_id}/results", json=bad)
        assert rejected.status_code == 422
        print("[PASS] relief with one anchor rejected:", rejected.get_json()["error"][:60])

        # Cover edge through the API.
        cover = client.post(f"/api/specimens/{tile_id}/metrology", json={
            "title": "HTTP cover edge", "mode": "cover",
            "relief_result_id": result["id"],
            "points": [[0.48, 0.469], [0.60, 0.469], [0.72, 0.469]],
        })
        assert cover.status_code == 201, cover.get_data(as_text=True)
        cover_record = cover.get_json()
        assert cover_record["mode"] == "cover"
        shape = cover_record["shape"]
        assert shape["sloped_length_mm_measured_only"] > 0
        assert shape["projected_length_mm"] >= shape["sloped_length_mm_measured_only"]
        print(f"[PASS] cover edge via HTTP: sloped {shape['sloped_length_mm_measured_only']} mm, "
              f"projected {shape['projected_length_mm']} mm, gaps {shape['gap_count']}")

        # Cover overlay and JSON artifacts.
        overlay = client.get(f"/api/metrology/{cover_record['id']}/overlay.png")
        assert overlay.status_code == 200 and overlay.mimetype == "image/png"
        doc = client.get(f"/api/metrology/{cover_record['id']}/data.json")
        assert doc.status_code == 200 and "sloped_length_mm_measured_only" in doc.get_data(as_text=True)
        print("[PASS] cover overlay PNG and data JSON served")

        # Wrong relief reference rejected.
        wrong = client.post(f"/api/specimens/{tile_id}/metrology", json={
            "title": "bad", "mode": "cover", "relief_result_id": make_id(),
            "points": [[0.1, 0.1], [0.2, 0.2]],
        })
        assert wrong.status_code in (404, 422)
        print("[PASS] cover edge with unknown relief reference rejected")

        # Trace mode still works (no regression).
        trace = client.post(f"/api/specimens/{tile_id}/metrology", json={
            "title": "plain trace", "mode": "trace",
            "points": [[0.1, 0.1], [0.2, 0.2]],
        })
        assert trace.status_code == 201, trace.get_data(as_text=True)
        print("[PASS] existing trace geometry still works")

    print("\nHTTP SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
