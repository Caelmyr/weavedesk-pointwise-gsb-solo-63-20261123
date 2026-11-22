import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weavedesk.web import create_app

data = Path(tempfile.mkdtemp())
app = create_app(data)
c = app.test_client()
c.post("/api/training", json={})
tile = c.get("/api/specimens/" + c.get("/api/specimens").get_json()["items"][0]["id"]).get_json()
base = next(p for p in tile["photos"] if p["registration_state"] == "base")
second = next(p for p in tile["photos"] if p["registration_state"] == "registered")
region = next(r for r in tile["regions"] if r["name"] == "Ground weave window")

out = Path(__file__).resolve().parent / "artifacts"
out.mkdir(exist_ok=True)

res = c.post(f"/api/specimens/{tile['id']}/survey/preview",
             json={"survey_axis": 90, "pins": [[0.12, 0.5], [0.88, 0.5]]})
(out / "survey_preview.png").write_bytes(res.data)
print("survey preview header:", res.headers["X-Survey-Summary"][:400])

c.put(f"/api/specimens/{tile['id']}/survey",
      json={"version": tile["version"], "survey_axis": 90, "pins": [[0.12, 0.5], [0.88, 0.5]]})
res = c.post(f"/api/specimens/{tile['id']}/results", json={
    "kind": "weave", "photo_id": base["id"], "comparison_id": second["id"],
    "region_id": region["id"],
    "options": {"analysis_mode": "repeat", "segment_length": 42, "band_width": 14},
})
result = res.get_json()
overlay = c.get(f"/api/results/{result['id']}/image/overlay")
(out / "repeat_overlay.png").write_bytes(overlay.data)
m = result["metrics"]
print("segments:", m["measured_segments"], "measured,", m["blank_segments"], "blank")
print("region rows:", m["region_rows"])
print("written to", out)
