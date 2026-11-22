import {q, e, btn, option, img, api, request, stat, n} from "./widgets.js";

const state = {seams: [], active: null, specimens: [], scanRows: null, previewUrl: null};
let toastTimer;

function toast(message, isError = false) {
  const box = q("#toast");
  box.className = isError ? "error" : "";
  box.textContent = message;
  if (message) toastTimer = setTimeout(() => { box.textContent = ""; }, isError ? 8500 : 4200);
}

function photoOptions(specimen, selectedId) {
  const photos = specimen.photos || [];
  return photos
    .filter(photo => photo.width && photo.height)
    .map(photo => option(photo.id, `${photo.role_label || photo.role} · ${photo.name}`, selectedId))
    .join("");
}

function specimenCards() {
  if (!state.specimens.length) {
    return `<div class="empty"><p>Upload the two fragment photographs under a specimen first, then start the seam here.</p></div>`;
  }
  return state.specimens.map(tile => {
    const photos = tile.photos || [];
    const defaultRight = photos.length > 1 ? photos[1].id : (photos[0]?.id ?? "");
    return `
    <div class="card seam-specimen" data-specimen="${e(tile.id)}">
      <div class="title-row"><h3>${e(tile.name)}</h3></div>
      <p class="soft">${photos.length} capture(s) · ${e(tile.body || "woven band")}</p>
      <div class="seam-photo-picker">
        <label>Left fragment
          <select data-role="left">${photos.map(p => option(p.id, p.name)).join("")}</select>
        </label>
        <label>Right fragment
          <select data-role="right">${photos.map(p => option(p.id, p.name, defaultRight)).join("")}</select>
        </label>
        <label>Notch step (px)<input data-role="step" type="number" min="0.5" max="400" step="0.5" value="30"></label>
      </div>
      <p class="soft">Mark the four band corners and the torn outline on each fragment, then build the seam. Practice specimens created by the seam loader already carry exact marks.</p>
      <div class="actions">${btn("Build seam", "seam-build", "primary", "measure", `data-specimen="${e(tile.id)}"`)}</div>
    </div>`;
  }).join("");
}

function home() {
  return `
  <section class="hero seam-hero">
    <div class="hero-text">
      <h1>Rejoin a broken ribbon.</h1>
      <p>Choose the one legal notch that keeps the blue edge stripes continuous and joins the warp ends that genuinely meet. Pixels are never invented where the tear was not photographed, and every line-scan value points back to its original fragment.</p>
    </div>
  </section>
  <div class="section-head"><div><h2>Saved seams</h2><p>${state.seams.length} record(s)</p></div></div>
  <div class="collection" id="seam-list">${seamCards()}</div>
  <div class="section-head"><div><h2>Start from fragment captures</h2></div></div>
  <div class="stack">${specimenCards()}</div>`;
}

function seamCards() {
  if (!state.seams.length) return `<div class="empty"><p>No seam recorded yet. Load the practice fragments to see the workflow.</p></div>`;
  return state.seams.map(item => `
    <button class="tile-card" type="button" data-do="seam-open" data-id="${e(item.id)}">
      <div class="tile-cover">${img(`/api/seams/${item.id}/composite.png`, `Seam ${item.title}`)}</div>
      <div class="tile-info">
        <h3>${e(item.title)}</h3>
        <p>Chosen: ${e(item.chosen_label)} · shift ${n(item.shift_px, 1)} px</p>
      </div>
    </button>`).join("");
}

function notchRows(report) {
  return report.map(row => `
    <tr class="${row.notch === state.active.chosen_notch ? "chosen-row" : ""}">
      <td>${e(row.label)}${row.notch === state.active.chosen_notch ? " ✓" : ""}</td>
      <td>${n(row.score, 3)}</td>
      <td>${row.stripe_penalty == null ? "—" : n(row.stripe_penalty, 2)}</td>
      <td>${n(row.thread_penalty, 2)}</td>
      <td>${row.matched_thread_count}</td>
      <td>${btn("Preview", "notch-preview", "tiny secondary", null, `data-notch="${row.notch}"`)}</td>
    </tr>`).join("");
}

function evidenceTable(active) {
  const ev = active.evidence;
  const pairRows = active.notch_report.find(r => r.notch === active.chosen_notch).thread_matches;
  const pairSet = new Set(pairRows.map(p => `${p.left}|${p.right}`));
  const leftLabels = ev.left_threads.map(t => t.label);
  const rightLabels = ev.right_threads.map(t => t.label);
  const unmatchedL = active.notch_report.find(r => r.notch === active.chosen_notch).unmatched_left_threads;
  const unmatchedR = active.notch_report.find(r => r.notch === active.chosen_notch).unmatched_right_threads;
  return `
    <div class="grid-detail">
      <section class="card">
        <h3>Blue stripe groups</h3>
        <div class="table-wrap"><table><thead><tr><th>Fragment</th><th>Group</th><th>Band rows</th><th>Center y (px)</th></tr></thead>
        <tbody>
          ${ev.left_stripes.map(s => `<tr><td>Left</td><td>${e(s.label)}</td><td>${s.rows}–${s.row_end}</td><td>${n(s.y, 1)}</td></tr>`).join("")}
          ${ev.right_stripes.map(s => `<tr><td>Right</td><td>${e(s.label)}</td><td>${s.rows}–${s.row_end}</td><td>${n(s.y, 1)}</td></tr>`).join("")}
        </tbody></table></div>
      </section>
      <section class="card">
        <h3>Loose warp ends</h3>
        <p class="soft">Green pairings below are the ends that genuinely meet at the chosen notch. Orange-ringed ends in the image have no visible partner and are left unjoined.</p>
        <div class="metrics">
          ${stat("Left ends", ev.left_threads.length)}${stat("Right ends", ev.right_threads.length)}${stat("Joined pairs", pairRows.length)}
        </div>
        <div class="table-wrap"><table><thead><tr><th>Left end</th><th>y (px)</th><th>Right end</th><th>y (px)</th><th>Residual</th><th>Status</th></tr></thead>
        <tbody>
          ${pairRows.map(p => `<tr><td>${e(p.left)}</td><td>${n(p.left_y, 1)}</td><td>${e(p.right)}</td><td>${n(p.right_y_band, 1)}</td><td>${n(p.residual_px, 2)}</td><td class="ok">joined</td></tr>`).join("")}
          ${unmatchedL.map(label => {
            const t = ev.left_threads.find(x => x.label === label);
            return `<tr><td>${e(label)}</td><td>${n(t.y, 1)}</td><td>—</td><td>—</td><td>—</td><td class="warn">no visible join</td></tr>`;
          }).join("")}
          ${unmatchedR.map(label => {
            const t = ev.right_threads.find(x => x.label === label);
            return `<tr><td>—</td><td>—</td><td>${e(label)}</td><td>${n(t.y, 1)}</td><td>—</td><td class="warn">no visible join</td></tr>`;
          }).join("")}
        </tbody></table></div>
      </section>
    </div>`;
}

function detailPage() {
  const active = state.active;
  const imageUrl = state.previewUrl || `/api/seams/${active.id}/composite.png?t=${Date.now()}`;
  return `
  <div class="grid-detail">
    <section class="card workspace-viewer">
      <div class="title-row"><h3>${e(active.title)}</h3><span class="badge">${e(active.chosen_label)}</span></div>
      <figure class="stage seam-stage">${img(imageUrl, "Annotated seam composite")}</figure>
      <p class="stage-foot">The dark gap is material neither fragment photographed; it is intentionally left empty. Green lines join warp ends that meet; orange rings mark ends with no visible partner.</p>
      <div class="actions">
        <a class="button secondary" href="/api/seams/${active.id}/composite.png" download>Save seam PNG</a>
        ${btn("Back to all seams", "seam-home", "ghost")}
      </div>
    </section>
    <div class="stack">
      <section class="card">
        <h3>Notch decision</h3>
        <div class="table-wrap"><table><thead><tr><th>Placement</th><th>Total score</th><th>Stripe penalty</th><th>Warp penalty</th><th>Joins</th><th></th></tr></thead>
        <tbody>${notchRows(active.notch_report)}</tbody></table></div>
        <p class="soft">Lower score is better. Only three placements are legal: one grid up, the middle grid, or one grid down.</p>
      </section>
      <section class="card">
        <h3>Line scan across the seam</h3>
        <p class="soft">Draw endpoints as fractions of the stitched band (x along its length, y across its width). Stations over the empty gap return blank rows with no color.</p>
        <form id="scan-form" class="form-stack">
          <div class="point-group">
            <label>Start x<input name="sx" type="number" min="0" max="1" step="0.01" value="0.05"></label>
            <label>Start y<input name="sy" type="number" min="0" max="1" step="0.01" value="0.5"></label>
            <label>End x<input name="ex" type="number" min="0" max="1" step="0.01" value="0.95"></label>
            <label>End y<input name="ey" type="number" min="0" max="1" step="0.01" value="0.5"></label>
            <label>Stations<input name="stations" type="number" min="12" max="480" value="120"></label>
            <label>Band px<input name="band" type="number" min="0" max="24" step="1" value="3"></label>
          </div>
          <button class="button primary" type="submit">Run line scan</button>
        </form>
        <div id="scan-result"></div>
      </section>
    </div>
  </div>
  ${evidenceTable(active)}`;
}

function render() {
  q("#main").innerHTML = state.active ? detailPage() : home();
}

async function refreshSeams() {
  state.seams = (await api("/api/seams")).items;
}

function scanPlot(rows) {
  const width = 720, height = 180, pad = 36;
  const labs = rows.filter(r => r.lab).map(r => r.lab[0]);
  if (!labs.length) return `<p class="soft">The line never crossed a visible fragment.</p>`;
  const lo = Math.min(...labs), hi = Math.max(...labs);
  const x = i => pad + i * (width - pad * 2) / Math.max(1, rows.length - 1);
  const y = v => height - pad - (v - lo) * (height - pad * 2) / Math.max(0.001, hi - lo);
  let path = "", blanks = [];
  rows.forEach((r, i) => {
    if (!r.lab) { blanks.push(x(i)); return; }
    path += `${path ? "L" : "M"}${x(i).toFixed(1)},${y(r.lab[0]).toFixed(1)} `;
  });
  const blankMarks = blanks.map(cx => `<line x1="${cx}" y1="${pad - 8}" x2="${cx}" y2="${height - pad}" stroke="#e87c69" stroke-dasharray="3 3" stroke-width="1"/>`).join("");
  return `
    <svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Line scan lightness with blank gap stations">
      <path d="${path}" fill="none" stroke="#377360" stroke-width="2"/>${blankMarks}
      <text x="${pad}" y="14" fill="#5d6d60" font-size="10">CIE L* (dashed red = blank, unseen stations)</text>
      <text x="${pad}" y="${height - 8}" fill="#5d6d60" font-size="10">${n(lo, 1)}</text>
      <text x="${width - pad}" y="${height - 8}" text-anchor="end" fill="#5d6d60" font-size="10">${n(hi, 1)}</text>
    </svg>`;
}

function scanTable(rows) {
  return `<div class="table-wrap scan-table"><table><thead><tr><th>#</th><th>Fragment</th><th>L*</th><th>a*</th><th>b*</th><th>Source photo</th><th>Source pixel (x, y)</th></tr></thead>
  <tbody>${rows.map(r => `<tr class="${r.fragment ? "" : "blank-row"}">
    <td>${r.station}</td><td>${r.fragment ? e(r.fragment) : "blank"}</td>
    <td>${r.lab ? n(r.lab[0], 2) : "—"}</td><td>${r.lab ? n(r.lab[1], 2) : "—"}</td><td>${r.lab ? n(r.lab[2], 2) : "—"}</td>
    <td>${r.source_photo_id ? e(r.source_photo_id.slice(0, 10)) : "—"}</td>
    <td>${r.source_pixel ? r.source_pixel.map(v => n(v, 1)).join(", ") : "—"}</td></tr>`).join("")}</tbody></table></div>`;
}

document.addEventListener("click", async event => {
  const target = event.target.closest("[data-do]");
  if (!target) return;
  const action = target.dataset.do;
  try {
    if (action === "seam-open") {
      state.active = await api(`/api/seams/${target.dataset.id}`);
      state.previewUrl = null;
      state.scanRows = null;
      render();
    } else if (action === "seam-home") {
      state.active = null;
      await refreshSeams();
      render();
    } else if (action === "notch-preview") {
      const res = await request(`/api/seams/${state.active.id}/notch-preview`, {
        method: "POST", body: {notch: Number(target.dataset.notch)},
      });
      const blob = await res.blob();
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = URL.createObjectURL(blob);
      render();
    }
  } catch (err) { toast(err.message, true); }
});

document.addEventListener("submit", async event => {
  if (event.target.id !== "scan-form") return;
  event.preventDefault();
  const data = new FormData(event.target);
  try {
    const body = {
      start: [Number(data.get("sx")), Number(data.get("sy"))],
      end: [Number(data.get("ex")), Number(data.get("ey"))],
      stations: Number(data.get("stations")),
      band_pixels: Number(data.get("band")),
    };
    const result = await api(`/api/seams/${state.active.id}/line-scan`, {method: "POST", body});
    state.scanRows = result.scan.rows;
    q("#scan-result").innerHTML = scanPlot(state.scanRows) + scanTable(state.scanRows) +
      `<p class="soft">${result.scan.supported_stations} measured · ${result.scan.blank_stations} blank · ${e(result.scan.note)}</p>`;
    toast("Line scan saved with source-pixel provenance.");
  } catch (err) { toast(err.message, true); }
});

q("#load-demo").addEventListener("click", async () => {
  try {
    const seam = await api("/api/seams/demo", {method: "POST"});
    await refreshSeams();
    state.active = seam;
    render();
    toast("Practice seam built from the synthetic fragment pair.");
  } catch (err) { toast(err.message, true); }
});

// Building a seam from chosen specimen captures.
document.addEventListener("click", async event => {
  const build = event.target.closest('[data-do="seam-build"]');
  if (!build) return;
  const card = build.closest(".seam-specimen");
  const specimenId = build.dataset.specimen;
  const specimen = state.specimens.find(item => item.id === specimenId);
  const leftId = card.querySelector('[data-role="left"]').value;
  const rightId = card.querySelector('[data-role="right"]').value;
  const step = Number(card.querySelector('[data-role="step"]').value) || 30;
  if (leftId === rightId) { toast("Choose two different fragment photographs.", true); return; }
  const findPhoto = id => specimen.photos.find(p => p.id === id);
  const fragmentSpec = photo => {
    // Default: full-frame band corners; the synthetic demo carries exact corners.
    const corners = photo.seam_corners || [[0.05, 0.3], [0.95, 0.3], [0.95, 0.7], [0.05, 0.7]];
    const outline = photo.seam_outline || [
      corners[0], [0.5, corners[0][1]], [0.5, corners[2][1]], corners[3],
    ];
    return {photo_id: photo.id, corners, outline, pixels_per_mm: photo.pixels_per_mm || 4};
  };
  try {
    const seam = await api("/api/seams", {
      method: "POST",
      body: {
        title: `Seam · ${specimen.name}`,
        notch_step_px: step,
        left: fragmentSpec(findPhoto(leftId)),
        right: fragmentSpec(findPhoto(rightId)),
      },
    });
    await refreshSeams();
    state.active = seam;
    render();
    toast("Seam built.");
  } catch (err) { toast(err.message, true); }
});

async function boot() {
  try {
    const seams = await api("/api/seams");
    state.seams = seams.items;
    const summaries = (await api("/api/specimens")).items;
    // The list endpoint returns summaries; fetch each full record so the
    // fragment photo pickers have real photo IDs and capture details.
    state.specimens = await Promise.all(
      summaries.map(item => api(`/api/specimens/${item.id}`).catch(() => item))
    );
    render();
  } catch (err) { toast(err.message, true); }
}
boot();
