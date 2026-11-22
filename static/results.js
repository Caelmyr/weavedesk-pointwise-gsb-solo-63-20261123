import {badge, btn, e, img, link, n, percent, plot, stat, table, date, dots} from "./widgets.js";

const colors = ["#58836b","#b26a4a","#818d40","#786f99"];
const layers = {overlay:"Measurement map",source:"Source plane",paired:"Paired plane",reference:"Reference image"};

export function metricSet(record) {
  const m = record.metrics;
  const rows = {
    color: () => [["Median L*",n(m.median_lab[0])],["Within-plane p90 ΔE",n(m.delta_e_from_tile_median["90"])],["Pixels measured",n(m.pixels,0)]],
    dots: () => [["Resolved knots",m.dot_count],["Median area (px²)",n(m.median_area_px,2)],["Resolved knot area",percent(m.ink_area_percent)]],
    weave: () => [["Family A pitch",m.axis_a_pitch==null?"—":`${n(m.axis_a_pitch,3)} ${m.unit}`],["Family B pitch",m.axis_b_pitch==null?"—":`${n(m.axis_b_pitch,3)} ${m.unit}`],["Crossing",m.crossing_angle==null?"—":`${n(m.crossing_angle,1)}°`]],
    coverage: () => [["Fabric",percent(m.percent.glazed)],["Mount",percent(m.percent.bare)],["Mixed",percent(m.percent.mixed)],["Unmeasured",percent(m.percent.unmeasured)]],
    defects: () => {
      const reviewed = Object.values(record.decisions);
      return [["Observed",m.total_observations],["Confirmed",reviewed.filter(item=>item.decision==="confirm").length],["Dismissed",reviewed.filter(item=>item.decision==="dismiss").length]];
    },
    glare: () => [["Color retained",percent(m.retained_percent)],["Light unreliable",percent(m.unreliable_percent)],["Glare hotspots",m.hotspot_count]],
    direction: () => [["Left-bright detail",percent(m.left_strong_percent)],["Right-bright detail",percent(m.right_strong_percent)],["Paired support",percent(m.support.overlap_percent)]],
    texture: () => [["Medium local roughness",n(m.grid_roughness["50"],5)],["Ridge axis",m.texture_orientation ? n(m.texture_orientation.ridge_angle_clockwise)+"°" : "—"],["Coherence",n(m.texture_orientation?.coherence,3)]],
    craze: () => [["Trace length",`${n(m.network_length)} ${m.length_unit}`],["Junctions",n(m.junctions,0)],["Closed cells",n(m.closed_cell_count,0)]],
    rim: () => [["Rim median ΔE",n(m.all_edges.median_from_center_delta_e)],["Rim pixels",n(m.all_edges.pixels,0)],["Interior L*",n(m.interior.median_lab[0])]],
    profile: () => [["Distance",`${n(m.line_length)} ${m.distance_unit}`],["Valid stations",`${m.supported_stations}/${m.requested_stations}`],["Median L*",n(m.features.median_lightness)]],
    reference: () => [["Specimen to reference ΔE",n(m.delta_e76)],["Both visible & reliable",percent(m.judged_percent)],["Noticeable change",percent(m.noticeable_change_percent)]],
    marking: () => [["Measured stations",percent(m.observed_percent)],["Rule among measured",percent(m.midrib_percent_of_observed)],["Runs",m.runs.length]],
    joint: () => [["Measured stations",m.measured_stations],["Unmeasured stations",m.unmeasured_stations],["Median absolute contrast",n(m.median_absolute_step,4)]],
    blade: () => [["Marked axis",`${n(m.axis_length)} ${m.length_unit}`],["Largest visible width",m.maximum_width==null?"—":`${n(m.maximum_width)} ${m.length_unit}`],["Supported sections",`${m.observed_stations}/${m.requested_stations}`]],
    aggregate: () => [["Bright clusters",m.candidate_clusters],["Supported pixels",m.observed_pixels],["Bright fraction",percent(m.bright_fraction_of_support)]],
  }[record.kind]();
  return `<div class="metrics">${rows.map(([name,value])=>stat(name,value)).join("")}</div>`;
}

function legend(record) {
  if (!record.legend) return "";
  const scale=record.legend;
  if (scale.mode==="categorical") {
    return `<div class="legend"><span>Image classification</span><div class="swatches">${scale.labels.map(label=>dots(label.name,label.color)).join("")}</div></div>`;
  }
  return `<div class="legend"><span>${e(scale.label)}</span><span>${n(scale.low,4)} to ${n(scale.high,4)}</span></div><div class="gradient ${e(scale.mode)}"></div>`;
}

function colorDetail(m) {
  return `<h3>Color palette</h3><div class="swatches">${m.palette.map(item=>dots(percent(item.coverage_percent), item.swatch)).join(" ")}</div>
    <details><summary>Spatial cells</summary>${table(["Column, row","Pixels","Median Lab","From median ΔE"], m.cells.map(item=>[item.cell.join(", "),item.pixels,item.lab.map(val=>n(val)).join(" / "),n(item.delta_from_center)]))}</details>`;
}

function coverageDetail(m) {
  const series = ["glazed","mixed","bare"].map((topic,index)=>({values:m.depth_bands.map(item=>item.percent[topic]),color:colors[index]}));
  return `<h3>Coverage across the sample</h3>${plot(series,"Coverage from top to bottom",{min:0,max:100})}
  ${table(["Depth band","Fabric","Mixed","Mount","Unmeasured"],m.depth_bands.map(item=>[`${n(item.relative_top*100,0)}–${n(item.relative_bottom*100,0)}%`,...["glazed","mixed","bare","unmeasured"].map(key=>percent(item.percent[key]))]))}`;
}

function markingDetail(m) {
  const rows=m.runs.map(run=>[run.condition,n(run.first_station,1),n(run.last_station,1),run.stations,n(run.median_cross_width,1)]);
  return `<h3>Edge stitch visibility runs</h3><p class="plot-note">${e(m.note)}</p>
    ${table(["Condition",`From (${m.unit})`,`To (${m.unit})`,"Stations",`Median width (${m.unit})`],rows)}`;
}

function jointDetail(m) {
  const rows=m.stations.map(item=>[n(item.position,1),n(item.left,4),n(item.right,4),n(item.right_minus_left,4),item.status]);
  return `<h3>Contrast along the mounting seam</h3><p class="plot-note">${e(m.note)}</p>
    ${plot([{values:m.stations.map(item=>item.right_minus_left)}],"Right minus left lightness")}
    <details><summary>Station readings</summary>${table([`Distance (${m.unit})`,"Left","Right","Difference","Condition"],rows)}</details>`;
}

function aggregateDetail(m) {
  const rows=m.candidates.map(item=>[item.rank,item.registered_pixels,n(item.contrast,4),n(item.bounding_width,1),n(item.bounding_height,1)]);
  return `<h3>Pale fleck candidates</h3><p class="plot-note">${e(m.note)}</p>
    ${table(["Rank","Pixels","Peak contrast",`Width (${m.unit})`,`Height (${m.unit})`],rows)}`;
}

function bladeDetail(m) {
  const columns=[`Distance (${m.length_unit})`,`Width (${m.length_unit})`,"Cross-section support"];
  const rows=m.stations.map(station=>[n(station.position,2),n(station.width,2),station.state]);
  return `<h3>Binding widths</h3><p>${e(m.note)}</p>
    ${plot([{values:m.stations.map(station=>station.width)}],"Width along the marked axis")}
    <details><summary>Cross-section readings</summary>${table(columns,rows)}</details>`;
}

function defectsDetail(record) {
  const markers=record.metrics.observations;
  if (!markers.length) return "<p>No spots exceeded this run's current thresholds.</p>";
  return `<h3>Review local observations</h3>
    <div class="divider-list scroll-list">${markers.map(marker=>{
      const decision=record.decisions[marker.key]?.decision || "review";
      return `<div class="row"><div class="grow"><strong>${e(marker.key)} · ${e(marker.type)}</strong><br><small>${n(marker.area_pixels,0)} px · ${e(marker.polarity)} · ${e(decision)}</small></div>
        ${btn("Confirm","defect-choice","tiny secondary",null,`data-marker="${e(marker.key)}" data-value="confirm"`)}
        ${btn("Dismiss","defect-choice","tiny ghost",null,`data-marker="${e(marker.key)}" data-value="dismiss"`)}
        ${btn("Note","defect-note","tiny ghost",null,`data-marker="${e(marker.key)}"`)}</div>`;
    }).join("")}</div>`;
}

function textureDetail(m) {
  return `<h3>Texture repeats by direction</h3>
    ${plot(m.lag_curves.map((dir,index)=>({values:dir.points.map(point=>point.correlation),color:colors[index]})),"Spacing along each direction",{min:-1,max:1})}
    <div class="swatches">${m.lag_curves.map((dir,index)=>dots(dir.direction,colors[index])).join(" ")}</div>
    ${table(["Direction","Peak spacing","Correlation"],m.lag_curves.flatMap(dir=>dir.local_highs.map(point=>[dir.direction,point.spacing_mm ? `${n(point.spacing_mm)} mm` : `${n(point.spacing_px)} px`,n(point.correlation,3)])))}
    <p class="plot-note">${e(m.note)}</p>`;
}

function profileDetail(m) {
  return `<h3>Lightness along the marked line</h3>${plot([{values:m.rows.map(item=>item.lightness)}], "Lightness along the station band",{axis:`0–${n(m.line_length)} ${m.distance_unit}`})}
    <details><summary>Sample station values</summary>${table(["Distance","Valid samples","L*","a*","b*"],m.rows.map(item=>[`${n(item.position,3)} ${item.position_unit}`,item.samples,n(item.lightness),n(item.a),n(item.b)]))}</details>`;
}

function glareDetail(m) {
  return `<h3>Glare hotspots</h3><p class="plot-note">${e(m.note)}</p>
    <div class="metrics">${stat("Color retained",percent(m.retained_percent))}${stat("Light unreliable",percent(m.unreliable_percent))}${stat("Too little detail",percent(m.too_little_detail_percent))}${stat("Not visible",percent(m.not_visible_percent))}</div>
    ${table(["Hotspot","Pixels","Area mm²"],m.hotspots.map((item,index)=>[index+1,item.pixels,n(item.area_mm2,3)]))}`;
}

function directionDetail(m) {
  return `<h3>Raking contrast by cell</h3><p class="plot-note">${e(m.note)}</p>
    ${table(["Column, row","Signed median","Strong contrast"],m.cells.map(item=>[item.cell.join(", "),n(item.median_signed,4),percent(item.strong_percent)]))}`;
}

function crazeDetail(m) {
  return `<h3>Closed lines</h3><p class="plot-note">${e(m.note)}</p>
  ${table(["Cell","Pixels","Area mm²"],m.largest_cells.map((item,index)=>[index+1,item.pixels,n(item.mm2,3)]))}`;
}

function rimDetail(m) {
  return `<h3>Rim color by side</h3><p class="plot-note">${e(m.note)}</p>
  ${table(["Side","Pixels","Median from interior ΔE","Excluded"],m.sides.map(item=>[item.side,item.stats?.pixels,n(item.stats?.median_from_center_delta_e),percent(item.excluded_percent)]))}`;
}

function referenceDetail(m) {
  const parts = `<h3>${e(m.reference_title)}</h3>
    <div class="metrics">${stat("Specimen",m.specimen.swatch)}${stat("Approved reference",m.reference.swatch)}</div>
    <p class="notice">${e(m.capture_note)}</p>`;
  if (m.judged_pixels == null) return parts;
  return `${parts}
    <h3>Two-session re-check</h3><p class="plot-note">${e(m.note)}</p>
    <div class="metrics">${stat("Both visible & reliable",percent(m.judged_percent))}${stat("Noticeable change",percent(m.noticeable_change_percent))}${stat("Position median ΔE",n(m.position_delta_e_median,2))}</div>
    ${table(["Withheld at this position","Share of selected area"],[
      ["Old photo left blank",percent(m.old_left_blank_percent)],
      ["Light unreliable",percent(m.light_unreliable_percent)],
      ["Too little detail",percent(m.too_little_detail_percent)],
      ["Outside approved sample",percent(m.outside_old_region_percent)],
    ])}`;
}

function details(record) {
  const render = {
    weave:m=>`<h3>Visible yarn families</h3><p class="plot-note">${e(m.note)}</p>${table(["Family","Direction (°)",`Pitch (${m.unit})`,"Peak / background"],m.rows.map(item=>[item.family,n(item.angle_degrees,2),n(item.pitch,3),n(item.peak_to_background,2)]))}`,
    dots:m=>`<h3>Isolated yarn knots</h3><p class="plot-note">${e(m.note)}</p>${table(["Knot","Area (px²)","Area (mm²)","Major/minor","Angle (°)"],m.rows.map(r=>[
      r.dot, n(r.area_px,0), n(r.area_mm2,5), n(r.axis_ratio,2), n(r.major_angle_deg,1)
    ]))}`,
    color:colorDetail,coverage:coverageDetail,defects:defectsDetail,glare:glareDetail,
    direction:directionDetail,texture:textureDetail,profile:profileDetail,reference:referenceDetail,craze:crazeDetail,rim:rimDetail,
    marking:markingDetail,joint:jointDetail,aggregate:aggregateDetail,blade:bladeDetail,
  }[record.kind];
  return render(record.kind==="defects" ? record : record.metrics);
}

export function resultCard(record, layer, currentRevision) {
  const path=`/api/results/${record.id}`;
  const available = ["overlay","source",...(record.pictures.paired ? ["paired"] : []),...(record.metrics.reference_image ? ["reference"] : [])];
  const old = record.source_version !== currentRevision;
  return `<section class="card stack">
    <div class="title-row"><div><h3>${e(record.label)}</h3><small>Specimen revision ${record.source_version} · ${e(date(record.created_at))}</small></div>${old ? badge("Earlier specimen revision","caution") : badge("Saved measurement","ready")}</div>
    <div class="actions">${available.map(key=>btn(layers[key],"result-layer",layer===key?"tiny primary":"tiny secondary",null,`data-value="${key}"`)).join("")}</div>
    <figure class="stage">${img(path+"/image/"+layer+"?v="+record.version,`${record.label} · ${layers[layer]}`)}</figure>
    ${layer==="overlay"?legend(record):""}
    ${metricSet(record)}
    <section>${details(record)}</section>
    <details><summary>Measurement source</summary>${table(["Property","Value"],[
      ["Image",record.provenance.photo.display_name],["Image role",record.provenance.photo.role],["Region",record.provenance.region?.name || "Entire plane"],
      ["Photo digest",record.provenance.photo.original_digest],
      ["Supported area",percent(record.support.available_percent)],
      ["Clipped pixels",record.support.clipped_pixels],
      ["Neutral correction",record.provenance.neutral_card?.before_swatch || "Uncorrected"],
    ])}</details>
    <div class="actions">${link("Metrics CSV",path+"/metrics.csv","secondary","download")}${btn("Pin evidence","pin", "primary","archive")}</div>
  </section>`;
}

export function specimenResultRows(items) {
  if (!items.length) return "<p class='soft'>Run a measurement to start this specimen’s history.</p>";
  return `<div class="divider-list scroll-list">${items.map(item=>`<div class="row"><div class="grow"><strong>${e(item.label)}</strong><br><small>Revision ${item.source_version} · ${e(date(item.created_at))}</small></div>${btn("Open","result-open","tiny secondary",null,`data-id="${item.id}"`)}</div>`).join("")}</div>`;
}
