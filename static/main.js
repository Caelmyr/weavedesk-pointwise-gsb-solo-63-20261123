import {q,qa,icon,e,btn,link,badge,empty,option,img,field,textField,selectField,numField,checkField,areaField,api,request,date,n,percent,plot,stat,table} from "./widgets.js";
import {resultCard,specimenResultRows} from "./results.js";

const state={page:"collection",tile:null,tiles:[],tab:"captures",tool:"color",capture:null,viewer:"raw",layer:"overlay",result:null,config:null,results:[],refs:[],packets:[],study:null,studies:[],query:""};
const dialog=q("#dialog");
const nav=[
  ["collection","Specimens","tiles"],
  ["references","Reference library","book"],
  ["studies","Fabric comparisons","plot"],
  ["received","Received evidence","archive"],
];
const tabs=[
  ["captures","Captures","camera"],["regions","Sampling regions","frame"],["measure","Measure","measure"],["geometry","Geometry","frame"],["evidence","Evidence","archive"],
];
const tileUrl=()=>`/api/specimens/${state.tile.id}`;
const tileData=extra=>({version:state.tile.version,...extra});
let toastTimer;
let working=0;
let travel=0;
let dialogToken=0;
let drawPoints=[];
let drawLimit=0;
let currentForm=null;
let reviewUploads=[];

function toast(message, isError=false) {
  clearTimeout(toastTimer);
  const box=q("#toast");
  box.className=isError?"error":"";
  box.textContent=message;
  if (message) toastTimer=setTimeout(()=>{box.textContent="";},isError?8500:4200);
}

function chrome() {
  q("#rail").innerHTML=`<a href="#collection" class="brand"><span class="mark">${icon("logo")}</span><span><strong>WeaveDesk</strong><small>Textile inspection</small></span></a>
    <p class="nav-label">Studio</p><nav aria-label="Studio navigation">
    ${nav.map(([key,label,glyph])=>`<a class="nav-link ${(state.page==="tile"?"collection":state.page)===key?"active":""}" href="#${key}">${icon(glyph)}${e(label)}</a>`).join("")}</nav>
    <footer><span class="status-mark"></span>Local workspace</footer>`;
  const crumb=state.page==="tile"
    ? `<a href="#collection">Specimens</a><span aria-hidden="true">/</span><strong>${e(state.tile?.name || "Specimen")}</strong>`
    : `<strong>${e(nav.find(([key])=>key===state.page)?.[1]||"Specimens")}</strong>`;
  q("#mast").innerHTML=`<div class="breadcrumb">${crumb}</div><div class="actions">${state.page==="tile" ? btn("Add photo","photo-add","primary","plus") : btn("New specimen","tile-new","primary","plus")}</div>`;
}

function cardTiles() {
  const search=state.query.trim().toLocaleLowerCase();
  const chosen=state.tiles.filter(item=>[item.name,item.glaze,item.firing,item.body,item.batch].some(value=>String(value).toLocaleLowerCase().includes(search)));
  if (!chosen.length) return state.tiles.length?empty("No specimen matches this search."):empty("Your collection is empty.",btn("Load textile samples","training","secondary","sample"));
  return `<div class="collection">${chosen.map(item=>`<button class="tile-card" type="button" data-do="tile-open" data-id="${item.id}">
    <div class="tile-cover">${item.cover_url?img(item.cover_url,`${item.name} photograph`, 'loading="lazy"'):icon("camera")}${item.synthetic?badge("Synthetic","training"):""}</div>
    <div class="tile-info"><h3>${e(item.name)}</h3><p>${e([item.glaze,item.body].filter(Boolean).join(" · ") || "Open to add a photo")}</p>
    <div class="tile-meta"><span>${icon("camera")} ${item.photo_count} ${item.photo_count===1?"view":"views"}</span><span>${e(item.firing||item.batch||"—")}</span></div></div></button>`).join("")}</div>`;
}

function collection() {
  return `<section class="hero"><div class="hero-text">
      <h1>Inspect a textile sample.</h1>
      <p>Register textile photographs, inspect weave and wear, and retain the original evidence.</p>
      <div class="actions">${btn(state.tiles.some(record=>record.synthetic)?"Open textile samples":"Load textile samples","training","light","sample")}${btn("Add a specimen","tile-new","secondary","plus")}</div>
      </div><figure class="hero-image">${img("/api/training/hero.png","Synthetic interlaced textile sample under a loupe")}<figcaption class="hero-caption">Indigo sampler · A · Synthetic training photo</figcaption></figure></section>
    <div class="section-head"><div><h2>Mounted specimens</h2><p>${state.tiles.length} ${state.tiles.length===1?"specimen":"specimens"}</p></div><label class="filter">${icon("search")}<input id="tile-search" placeholder="Search specimens" aria-label="Search specimens" value="${e(state.query)}"></label></div>
    <div id="tile-list">${cardTiles()}</div>`;
}

function currentPhoto() {
  return state.tile?.photos.find(item=>item.id===state.capture) || state.tile?.photos.find(item=>item.registration_state==="base") || state.tile?.photos[0];
}

function sources(selects, selected, optional) {
  return (optional?option("",optional,selected):"")+selects.map(item=>option(item.id,item.role_label+" · "+item.name,selected)).join("");
}

function regions(purpose,includeNone=true) {
  const choices=state.tile.regions.filter(item=>purpose?item.purpose===purpose:item.purpose!=="ignore");
  return (includeNone?option("","Entire plane",""):"")+choices.map(item=>option(item.id,item.name)).join("");
}

function imagesForCapture() {
  const photo=currentPhoto();
  const tile=state.tile;
  if (!photo) return empty("Add a diffuse photograph, then mark the corners of the survey.",btn("Add photo","photo-add","primary","plus"));
  const canvas=photo.registered_url && state.viewer!=="source";
  const illustration=canvas?photo.registered_url+`&layer=${state.viewer}`:photo.preview_url;
  const canModes=Boolean(photo.registered_url);
  return `<section class="card workspace-viewer">
      <div class="title-row"><h3>${e(photo.role_label)}</h3>${badge(photo.registration_state,canModes?"ready":"caution")}</div>
      <div class="actions">${[["source","Original framing"],["raw","Survey plane"],["regions","Regions"],...(tile.calibration?[["corrected","Neutral balance"]]:[])].filter(([key])=>canModes||key==="source").map(([key,label])=>btn(label,"viewer",state.viewer===key?"tiny primary":"tiny secondary",null,`data-value="${key}"`)).join("")}</div>
      <figure class="stage">${img(illustration,`${photo.role_label} · ${tile.name}`)}</figure>
      <p class="stage-foot">${photo.width} × ${photo.height} px · ${e(photo.working_color_space)}${tile.canvas_size&&canvas?` · Survey plane ${tile.canvas_size.join(" × ")} px`:""}</p>
      <div class="actions">${btn(photo.registration_state==="base"?"Mark survey corners":photo.registered_url?"Edit registration":"Register photo","geometry","secondary","frame")}
      ${photo.registration_state!=="base"?btn("Set as base","geometry-base","secondary"):""}
      ${tile.photos.filter(item=>item.registered_url).length>1?btn("Compare registered views","compare","secondary","compare"):""}
      ${canModes?btn("Image quality","quality","secondary","plot"):""}${link("Download original",photo.original_url,"secondary","download")}</div>
    </section>`;
}

function captureSide() {
  const tile=state.tile;
  const selected=currentPhoto();
  const rows=tile.photos.map(item=>`<div class="row">${img(item.thumbnail_url,"", 'class="item-image"')}
    <div class="grow"><span class="row-title">${e(item.role_label)}</span><small>${e(item.registration_state)} · ${item.width} × ${item.height}</small></div>
    ${btn("View","photo-view",selected?.id===item.id?"tiny primary":"tiny secondary",null,`data-id="${item.id}"`)}</div>`).join("");
  const listed=selected?selected.diagnostics.map(value=>`<li>${e(value)}</li>`).join(""):"";
  return `<div class="stack"><section class="card"><div class="title-row"><h3>Capture set</h3>${btn("Add photo","photo-add","tiny secondary","plus")}</div>
  ${rows?`<div class="divider-list">${rows}</div>`:"<p class='soft'>No photos in this set.</p>"}
  </section>${selected?`<section class="card"><h3>Capture observations</h3><div class="metrics">
  ${stat("Highlight clipping",percent(selected.stats.white_clip_percent))}${stat("Shadow clipping",percent(selected.stats.black_clip_percent))}${stat("Opaque support",percent(selected.stats.supported_percent))}</div>
  ${listed?`<ul>${listed}</ul>`:"<p class='soft'>No warnings at thumbnail scale.</p>"}<div class="actions">${btn("Edit photo","photo-edit","tiny secondary","edit")}${btn("Remove photo","photo-remove","tiny danger")}</div></section>`:""}</div>`;
}

function captures() {
  return `<div class="grid-detail">${imagesForCapture()}${captureSide()}</div>`;
}

function regionPage() {
  const tile=state.tile;
  const current=tile.photos.find(item=>item.registration_state==="base");
  const rows=tile.regions.map(item=>`<div class="row"><div class="grow"><strong class="row-title">${e(item.name)}</strong><small>${e(state.config.region_purposes[item.purpose])}</small></div>
    ${btn("Edit","region-edit","tiny secondary",null,`data-id="${item.id}"`)}</div>`).join("");
  const gains=tile.calibration;
  const line=tile.glare_line;
  const exampleRegions=tile.regions.filter(item=>item.purpose!=="ignore");
  const lineCard=`<section class="card"><h3>One polarized glare line</h3>
    ${line?`<div class="metrics">${stat("Film glare examples",line.examples.filter(item=>item.kind==="glare").length)}${stat("Retained yarn examples",line.examples.filter(item=>item.kind==="color").length)}${stat("Color kept in examples",percent(line.counts.example_color_retained_percent))}${stat("Glare rejected in examples",percent(100-line.counts.example_glare_retained_percent))}</div>
      <p class="soft">Adopted ${e(date(line.created_at))}. Every polarized run and reference re-check on this specimen reads this same line.</p>`:"<p class='soft'>Mark one confirmed film-glare patch and one confirmed bright-yarn patch, then adopt the specimen's single line. This one line is reused by all later polarised work.</p>"}
    <div class="actions">${btn(line?"Replace glare line":"Adopt glare line","glare-line","secondary","sample")}${line?btn("Clear line","glare-line-clear","ghost"):""}</div>
    ${exampleRegions.length<2?"<p class='soft'>Draw at least two sampling regions first.</p>":""}
  </section>`;
  return `<div class="grid-detail"><section class="card"><div class="title-row"><h3>Sampling plane</h3>${current?btn("Draw region","region-add","primary","plus"):""}</div>
     ${current?`<figure class="stage">${img(current.registered_url+"&layer=regions",`Marked regions on ${tile.name}`)}</figure><p class="stage-foot">Region positions are saved in the registered survey plane.</p>`:empty("Set a base photograph from Captures.")}</section>
     <div class="stack"><section class="card"><h3>Saved regions</h3>${rows?`<div class="divider-list">${rows}</div>`:"<p class='soft'>Draw an inspection area, a coverage reference, or a patch to exclude.</p>"}</section>
     ${lineCard}
     <section class="card"><h3>Neutral-card balance</h3>${gains?`<div class="metrics">${stat("Measured card",gains.before_swatch)}${stat("Balanced card",gains.after_swatch)}${stat("Unclipped card pixels",n(gains.card_samples,0))}</div>`:"<p class='soft'>Use an evenly lit neutral patch to balance this base capture.</p>"}
       <div class="actions">${btn("Balance neutral card","neutral","secondary","sample")}${gains?btn("Clear balance","neutral-clear","ghost"):""}</div>
     </section></div></div>`;
}

function optionInput(key,item) {
  const {kind,label,default:value}=item;
  if (kind==="number"||kind==="integer") return numField(label,key,value,item.min,item.max,item.step);
  if (kind==="choice") return selectField(label,key,item.choices.map(choice=>option(choice,choice,value)).join(""));
  if (kind==="flag") return checkField(label,key,value);
  if (kind==="region") return selectField(label,key,option("","Auto / current area","")+state.tile.regions.filter(record=>item.purpose?record.purpose===item.purpose:record.purpose!=="ignore").map(record=>option(record.id,record.name)).join(""));
  if (kind==="reference") return selectField(label,key,option("","Choose a reference","")+state.refs.map(record=>option(record.id,record.title)).join(""));
  if (kind==="point") return `<div class="field"><span>${e(label)} (X, Y)</span><div class="point-group">
    ${value.map((coordinate,index)=>`<input name="${key}-${index}" aria-label="${e(label)} ${index===0?"X":"Y"}" type="number" min="0" max="1" step=".001" value="${coordinate}">`).join("")}</div></div>`;
  return "";
}

function measureControls() {
  const tool=state.config.tools[state.tool];
  const paired=tool.capture_pair || (state.tool==="reference"&&state.tile.glare_line?["parallel","cross"]:null);
  const all=state.tile.photos.filter(item=>item.registered_url);
  const match=field=>sources(paired?all.filter(item=>item.role===paired[field]):all,null,paired?"Auto if unique":"Base capture");
  return `<form id="measurement-form" class="form-stack">
    ${selectField("Measurement","kind",Object.entries(state.config.tools).map(([name,item])=>option(name,item.label,state.tool)).join(""))}
    <p class="soft">${e(tool.description)}</p>
    ${selectField(paired?"First photo":"Photo source","photo_id",match(0))}
    ${paired?selectField("Paired photo","comparison_id",match(1)):checkField("Apply saved neutral balance","apply_neutral",true)}
    ${state.tool==="reference"&&!state.tile.glare_line?"<p class='soft'>This specimen has no adopted glare line, so Reference match compares the approved sample directly.</p>":""}
    ${selectField("Region","region_id",regions())}
    <div class="form-columns">${Object.entries(tool.fields).map(([key,item])=>optionInput(key,item)).join("")}</div>
    ${["profile","marking","joint","blade"].includes(state.tool)?btn("Mark line on image","mark-profile","secondary","measure"):""}
    <button class="button primary" type="submit">${icon("measure")}Run measurement</button>
  </form>`;
}

function measurements() {
  return `<div class="grid-detail"><div>${state.result?resultCard(state.result,state.layer,state.tile.version):empty("Run a measurement to view its map and results.")}</div>
    <div class="stack"><section class="card"><h3>Measure specimen</h3>${measureControls()}</section><section class="card"><h3>Saved measurements</h3>${specimenResultRows(state.results)}</section></div></div>`;
}

function evidencePage() {
  return `<div class="grid-detail"><section class="card"><div class="title-row"><h3>Pinned evidence</h3>${btn("Pin evidence","pin","primary","archive")}</div>
    ${state.packets.length?`<div class="divider-list">${state.packets.map(item=>`<div class="row"><div class="grow"><strong class="row-title">${e(item.title)}</strong><small>${item.count} measurements · ${e(date(item.created_at))}</small></div>
    ${link("View report",`/api/evidence/${item.id}/inspection.html`,"tiny secondary","book")}${link("ZIP",`/api/evidence/${item.id}/archive.zip`,"tiny secondary","download")}</div>`).join("")}</div>`:empty("Pin a packet from saved measurements to keep a dated copy of the observations.")}</section>
    <section class="card"><h3>Specimen activity</h3><div class="divider-list scroll-list">${(state.activity||[]).map(row=>`<div class="row"><div class="grow"><strong>${e(row.message)}</strong><br><small>${e(date(row.recorded_at))}</small></div></div>`).join("")}</div></section></div>`;
}

function geometryPage() {
  const record=state.metrologyActive;
  const results=state.metrology||[];
  const tile=state.tile;
  let report=empty("Save a traced path or the area of an existing sampling region.");
  if (record) {
    const metrics=record.shape;
    const base=`/api/metrology/${record.id}`;
    const stats=record.mode==="trace"
      ? [["Trace length",`${n(metrics.trace_length)} ${metrics.length_unit}`],["Supported stations",percent(metrics.overall_support)]]
      : [["Marked area",`${n(metrics.vector_outline_area)} ${metrics.units}`],["Visible area",`${n(metrics.raster_supported_area)} ${metrics.units}`],["Outline",`${n(metrics.outline_length)} ${metrics.outline_unit}`]];
    const stations=record.mode==="trace"?metrics.segments:metrics.outline_segments;
    report=`<section class="card stack"><h3>${e(record.title)}</h3>
      <figure class="stage">${img(base+"/overlay.png",record.title)}</figure>
      <div class="metrics">${stats.map(([name,value])=>stat(name,value)).join("")}</div>
      ${table(["Segment","Length","Heading"],stations.map(item=>[item.index,`${n(item.length)} ${item.units}`,n(item.clockwise_heading)+"°"]))}
      <div class="actions">${link("Measurement JSON",base+"/data.json","secondary","download")}</div></section>`;
  }
  return `<div class="grid-detail">${report}<div class="stack"><section class="card"><h3>Measure a visible shape</h3>
      <p class="soft">Trace along the image or measure visible support inside a saved sampling region.</p>
      <div class="actions">${btn("Trace a path","trace-add","primary","measure")}${btn("Measure region area","area-add","secondary","frame")}</div></section>
      <section class="card"><h3>Saved geometry</h3><div class="divider-list">${results.map(item=>`<div class="row"><div class="grow">${e(item.title)}<br><small>${e(item.mode)} · revision ${item.specimen.version}</small></div>${btn("Open","metrology-open","tiny secondary",null,`data-id="${item.id}"`)}</div>`).join("")||"<p class='soft'>No saved outlines.</p>"}</div></section></div></div>`;
}

function receivedPage() {
  const items=state.received||[];
  const chosen=state.receivedActive;
  const actions=btn("Verify evidence ZIP","receive-add","primary","archive");
  let view=empty("Open an inspection ZIP to check its declared images and specimen details.",actions);
  if (chosen) {
    const path=`/api/received/${chosen.id}`;
    view=`<section class="card stack"><div class="title-row"><h3>${e(chosen.title)}</h3>${badge("Digests match","ready")}</div>
      <p>${e(chosen.specimen)} · revision ${chosen.specimen_revision}</p>
      ${chosen.notes?`<p class="notice">${e(chosen.notes)}</p>`:""}
      ${chosen.sheets.map((item,index)=>img(path+`/sheet/${index+1}`,`Received evidence sheet ${index+1}`)).join("")}
      ${table(["Measurement","Source capture","Version"],chosen.measurements.map(item=>[item.label,item.capture.name,item.source_version]))}
      <p class="soft">Archive matches its included digests. Preserve sender and capture records when sharing it.</p></section>`;
  }
  return `<header class="page-head"><div><h1>Received evidence</h1><p>Open a received WeaveDesk inspection packet and verify its declared images.</p></div>${actions}</header>
    <div class="grid-detail">${view}<section class="card"><h3>Received packets</h3><div class="divider-list">${items.map(item=>`<div class="row"><div class="grow">${e(item.title)}<br><small>${e(date(item.imported_at))}</small></div>${btn("Open","received-open","tiny secondary",null,`data-id="${item.id}"`)}</div>`).join("")||"<p class='soft'>No received packets.</p>"}</div></section></div>`;
}

function tilePage() {
  const tile=state.tile;
  const descriptors=[tile.glaze,tile.body,tile.firing,tile.width_mm&&tile.height_mm?`${tile.width_mm} × ${tile.height_mm} mm`:null].filter(Boolean);
  return `<header class="page-head"><div><p class="eyebrow">${tile.synthetic?"Synthetic training specimen":"Specimen"}</p><h1>${e(tile.name)}</h1>
  <p>${e(descriptors.join(" · ") || "Registered textile sample")}</p></div><div class="actions">${btn("Edit specimen","tile-edit","secondary","edit")}
  ${btn("Approve reference","reference-add","secondary","book")}</div></header>
  <nav class="tabs" aria-label="Specimen tools">${tabs.map(([key,label,glyph])=>`<button type="button" class="tab ${state.tab===key?"active":""}" data-do="tile-tab" data-value="${key}">${icon(glyph)}${e(label)}</button>`).join("")}</nav>
  ${{captures,regions:regionPage,measure:measurements,geometry:geometryPage,evidence:evidencePage}[state.tab]()}`;
}

function libraryPage() {
  const items=state.refs;
  return `<header class="page-head"><div><h1>Reference library</h1><p>Approved color samples retain their capture, region, and neutral-card basis.</p></div></header>
    ${items.length?`<div class="collection">${items.map(ref=>`<article class="card stack">${img(`/api/library/${ref.id}/image`,ref.title,'loading="lazy"')}
    <h3>${e(ref.title)}</h3><p>${e(ref.family || ref.specimen.name)} · ${e(ref.fingerprint.swatch)}</p>
    <div class="actions">${link("Open specimen","#tile:"+ref.tile_id,"secondary","tiles")}${btn("Retire","reference-retire","ghost",null,`data-id="${ref.id}"`)}</div></article>`).join("")}</div>`:empty("Approve a visual reference from a specimen with a registered capture.",link("Open specimens","#collection","primary","tiles"))}`;
}

function studyPanel() {
  const report=state.study;
  if (!report) return empty("Select specimens with saved measurements, then build a study.");
  const base=`/api/studies/${report.id}`;
  return `<section class="card stack"><h3>${e(report.title)}</h3><p>${e(report.comparison_note)}</p>
    <div class="metrics">${report.groups.map(item=>stat(`${item.group} · n=${item.n}`,n(item.mean,4))).join("")}</div>
    ${report.pictures.map((_,index)=>img(base+`/sheet/${index+1}`,`Survey comparison sheet ${index+1}`)).join("")}
    ${table(["Specimen","Group",report.metric_label,"Support"],report.rows.map(item=>[item.tile_name,item.group,n(item.value,4),percent(item.support)]))}
    ${report.caveats.map(item=>`<p class="notice">${e(item)}</p>`).join("")}<div class="actions">${link("CSV",base+"/data.csv","secondary","download")}${link("Study JSON",base+"/data.json","secondary","download")}</div></section>`;
}

function studyPage() {
  return `<header class="page-head"><div><h1>Fabric comparisons</h1><p>Compare retained measurements by series, mount condition, imaging visit or tray.</p></div>${btn("Build study","study-add","primary","plus")}</header>
    <div class="grid-detail">${studyPanel()}<section class="card"><h3>Saved studies</h3>
      ${state.studies.map(item=>`<div class="row"><div class="grow"><strong>${e(item.title)}</strong><br><small>${e(item.metric_label)}</small></div>${btn("Open","study-open","tiny secondary",null,`data-id="${item.id}"`)}</div>`).join("")||"<p class='soft'>No saved studies.</p>"}</section></div>`;
}

function render() {
  chrome();
  q("#main").innerHTML={collection,tile:tilePage,references:libraryPage,studies:studyPage,received:receivedPage}[state.page]();
}

async function gatherTile(token=travel) {
  const url=tileUrl();
  const [results,packets,activity,refs,metrology]=await Promise.all([
    api(url+"/results"),api(url+"/evidence"),api(url+"/activity"),api("/api/library"),api(url+"/metrology"),
  ]);
  if (token!==travel) return;
  state.results=results.items;
  state.packets=packets.items;
  state.activity=activity.items;
  state.metrology=metrology.items;
  state.refs=refs.items;
}

async function route() {
  const token=++travel;
  const hash=location.hash.substring(1);
  if (hash.startsWith("tile:")) {
    const id=hash.slice(5);
    if (!/^[a-f0-9]{24}$/.test(id)) throw Error("This specimen link is invalid.");
    const tile=await api(`/api/specimens/${id}`);
    if (token!==travel) return;
    if (state.tile?.id!==tile.id || state.page!=="tile") {
      state.tab="captures"; state.capture=null; state.result=null;state.layer="overlay";state.viewer="raw";state.metrologyActive=null;
    }
    state.tile=tile;
    await gatherTile(token);
    if (token===travel) {state.page="tile";render();}
    return;
  }
  if (hash==="references") {
    const refs=await api("/api/library");
    if (token===travel) {state.refs=refs.items;state.page="references";render();}
    return;
  }
  if (hash==="studies") {
    const runs=await api("/api/studies");
    if (token===travel) {state.studies=runs.items;state.page="studies";render();}
    return;
  }
  if (hash==="received") {
    const received=await api("/api/received");
    if (token===travel) {state.received=received.items;state.page="received";render();}
    return;
  }
  const collection=await api("/api/specimens");
  if (token===travel) {state.tiles=collection.items;state.page="collection";render();}
}

function modal(title, contents, submitLabel, onSubmit, wide=false, extra="") {
  ++dialogToken;
  qa("[data-blob]",dialog).forEach(el=>URL.revokeObjectURL(el.dataset.blob));
  dialog.className=wide?"wide":"";
  dialog.innerHTML=`<div class="dialog-head"><h2>${e(title)}</h2>${btn("Close","close","tiny ghost","x")}</div>
    <form id="dialog-form" class="form-stack"><div class="dialog-error" role="alert"></div>${contents}
      ${submitLabel?`<div class="actions dialog-controls">${extra}<button type="submit" class="button primary">${e(submitLabel)}</button></div>`:""}</form>`;
  currentForm=onSubmit;
  if (!dialog.open) dialog.showModal();
}

function commitModal() {
  ++dialogToken;
  if (dialog.open) dialog.close();
  currentForm=null;
}

function formData() {
  return new FormData(q("#dialog-form"));
}

async function patchTile(path, method, body, message, options={}) {
  const incoming=await api(tileUrl()+path,{method,body:tileData(body)});
  state.tile=incoming;
  await gatherTile();
  if (!options.keepModal) commitModal();
  toast(message);
  if (options.viewer) state.viewer=options.viewer;
  render();
}

function metadataForm(tile) {
  const values=tile||{};
  return `<div class="form-columns">
    ${textField("Specimen name","name",values.name,'required maxlength="120"')}
    ${textField("Mount condition","glaze",values.glaze,'maxlength="100"')}
    ${textField("Fabric group","body",values.body,'maxlength="100"')}
    ${textField("Imaging visit","firing",values.firing,'maxlength="100"')}
    ${textField("Tray","batch",values.batch,'maxlength="80"')}
    <div></div>${numField("Width mm","width_mm",values.width_mm,0.1,100000,0.1)}
    ${numField("Height mm","height_mm",values.height_mm,0.1,100000,0.1)}
    </div>${areaField("Notes","notes",values.notes)}`;
}

function plainData(data) {
  return Object.fromEntries(data.entries());
}

function parseDatesAndNumbers(data, fields) {
  const payload=plainData(data);
  for (const field of fields) payload[field]=payload[field] ? Number(payload[field]) : null;
  return payload;
}

function tileForm(existing) {
  modal(existing?"Edit specimen":"New specimen",metadataForm(existing),existing?"Save details":"Create specimen", async data=>{
    const payload=parseDatesAndNumbers(data,["width_mm","height_mm"]);
    if (existing) {
      await patchTile("","PATCH",payload,"Specimen details saved.");
      return;
    }
    const made=await api("/api/specimens",{method:"POST",body:payload});
    commitModal();
    location.hash="tile:"+made.id;
    toast("Specimen created.");
  });
}

function captureForm(photo) {
  const available=Object.entries(state.config.photo_roles).map(([key,label])=>option(key,label,photo?.role||"diffuse")).join("");
  modal(photo?"Edit capture":"Add a capture",`
    ${photo?"":field("Photo","photo",'<input id="field-photo" name="photo" type="file" accept="image/png,image/jpeg,image/tiff,image/webp,.tif,.tiff" required>')}
    ${selectField("Lighting role","role",available)}${areaField("Capture notes","note",photo?.note,500)}
    ${photo?"":'<p class="soft">PNG, JPEG, WebP, or single-frame TIFF up to 24 MiB. The original is retained.</p>'}`,
    photo?"Save capture":"Import photo",async data=>{
      if (photo) {
        await patchTile("/photos/"+photo.id,"PATCH",plainData(data),"Capture updated.");
        return;
      }
      data.set("version",state.tile.version);
      const updated=await api(tileUrl()+"/photos",{method:"POST",body:data});
      state.tile=updated;
      state.capture=state.tile.photos.at(-1).id;
      state.viewer="raw";
      await gatherTile();
      commitModal();
      toast("Photo imported.");
      render();
    });
}

const drawing = (image,alt) => `<div class="stage"><div class="capture-drawing" id="draw-stage">${img(image,alt)}
  <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true" id="draw-overlay"></svg></div></div>`;

function drawUpdate() {
  const artist=q("#draw-overlay");
  if (!artist) return;
  const coords=drawPoints.map(([x,y])=>[x*1000,y*1000]);
  const edges=(drawLimit===2||coords.length<3)?"polyline":"polygon";
  artist.innerHTML=`<${edges} class="draw-line" points="${coords.map(pair=>pair.join(",")).join(" ")}"></${edges}>
    ${coords.map(([x,y],index)=>`<circle class="draw-point" cx="${x}" cy="${y}" r="10"/><text class="draw-label" x="${x+16}" y="${y+16}">${index+1}</text>`).join("")}`;
  const label=q("#draw-count");
  if (label) label.textContent=`${drawPoints.length} of ${drawLimit} points marked`;
}

function prepareDrawing(chosen, limit) {
  drawPoints=chosen.map(value=>[...value]);
  drawLimit=limit;
  drawUpdate();
}

function geometryForm(mode) {
  const item=currentPhoto();
  if (!item) throw Error("Select a photo to register.");
  const plane=state.tile.plane;
  const saving=mode==="base";
  const existing=saving?(plane?.photo_id===item.id?plane?.corners:[]):state.tile.registrations[item.id]?.corners;
  modal(saving?"Mark the specimen plane":"Register capture",`
    <p class="notice">Mark the same four points clockwise, beginning at the upper-left survey corner. Clear the marks before drawing a new outline.</p>
    ${drawing(item.preview_url,"Photo to mark")}<div class="actions">${btn("Clear points","draw-clear","tiny secondary")}${btn("Undo point","draw-undo","tiny ghost")}
      ${btn("Preview alignment","geometry-preview","tiny secondary",null,`data-mode="${mode}" data-id="${item.id}"`)}<span class="soft" id="draw-count"></span></div>
    ${saving?checkField("Reset existing regions and registrations if the plane changes","reset_regions",false):""}
    ${saving?checkField("Use entered specimen proportions","respect_scale",true):""}
    <div id="geometry-preview"></div>`,saving?"Save specimen plane":"Save registration",async data=>{
      if (drawPoints.length!==4) throw Error("Mark exactly four survey corners.");
      if (saving) {
        await patchTile("/plane","PUT",{
          photo_id:item.id,corners:drawPoints,reset_regions:data.has("reset_regions"),respect_scale:data.has("respect_scale"),
        },"Specimen plane saved.",{viewer:"raw"});
      } else {
        await patchTile("/registrations/"+item.id,"PUT",{corners:drawPoints},"Capture registered.",{viewer:"raw"});
      }
    },true);
  prepareDrawing(existing||[],4);
}

function regionForm(region) {
  const base=state.tile.photos.find(item=>item.registration_state==="base");
  if (!base) throw Error("Set a base capture before drawing a region.");
  const choices=Object.entries(state.config.region_purposes).map(([key,label])=>option(key,label,region?.purpose||"inspection")).join("");
  modal(region?"Edit sampling region":"Draw sampling region",`
    <div class="form-columns">${textField("Name","name",region?.name,'required maxlength="80"')}${selectField("Purpose","purpose",choices)}</div>
    <p class="notice">Click around the area in order. Use at least three points and leave a little room at capture edges.</p>
    ${drawing(base.registered_url,"Specimen plane for region drawing")}
    <div class="actions">${btn("Clear points","draw-clear","tiny secondary")}${btn("Undo point","draw-undo","tiny ghost")}<span class="soft" id="draw-count"></span></div>`,
    "Save region",async data=>{
      if (drawPoints.length<3) throw Error("Mark at least three points to close the region.");
      const fields={...plainData(data),vertices:drawPoints};
      const path=region?"/regions/"+region.id:"/regions";
      await patchTile(path,region?"PUT":"POST",fields,"Sampling region saved.");
    },true,region?btn("Remove region","region-remove","danger",null,`data-id="${region.id}"`):"");
  prepareDrawing(region?.vertices||[],40);
}

function lineForm() {
  const form=q("#measurement-form");
  const id=form.elements.photo_id.value;
  const base=state.tile.photos.find(item=>id ? item.id===id : item.registration_state==="base");
  if (!base) throw Error("Set a base capture before drawing a line.");
  const start=["start-0","start-1"].map(name=>Number(form.elements[name].value));
  const end=["end-0","end-1"].map(name=>Number(form.elements[name].value));
  modal("Mark a line scan",`<p class="notice">Mark the two ends of the line, then choose Use endpoints.</p>
    ${drawing(base.registered_url,"Specimen plane for line selection")}
    <div class="actions">${btn("Clear points","draw-clear","tiny secondary")}${btn("Undo point","draw-undo","tiny ghost")}<span class="soft" id="draw-count"></span></div>`,
    "Use endpoints",async ()=>{
      if (drawPoints.length!==2) throw Error("Mark two endpoints.");
      for (const [i,key] of ["start","end"].entries()) {
        for (const [j,value] of drawPoints[i].entries()) form.elements[`${key}-${j}`].value=value.toFixed(3);
      }
      commitModal();
    },true);
  prepareDrawing([start,end],2);
}

function neutralForm() {
  const available=state.tile.regions.filter(item=>item.purpose==="neutral");
  if (!available.length) throw Error("Draw a region marked Neutral color card first.");
  modal("Balance neutral card",selectField("Card region","region_id",available.map(item=>option(item.id,item.name)).join(""))+
    numField("Linear target","target",0.18,0.08,0.8,0.01)+
    '<p class="soft">Target is a linear-image value. Sampling uses the unclipped center of the marked card.</p>',
    "Save balance",async data=>patchTile("/calibration","PUT",parseDatesAndNumbers(data,["target"]),"Neutral-card balance saved."));
}

function refForm() {
  if (!state.tile.plane) throw Error("Add a specimen photo before approving a reference.");
  modal("Approve visual reference",`${textField("Reference name","title",state.tile.name,'required maxlength="100"')}
    ${textField("Reference series","family",state.tile.glaze,'maxlength="100"')}${selectField("Region","region_id",regions())}
    ${areaField("Notes","notes","",1200)}<p class="notice">The selected image and measured color remain attached to the reference if the specimen later changes.</p>`,
    "Approve reference",async data=>{
      await api("/api/library",{method:"POST",body:{...plainData(data),tile_id:state.tile.id}});
      await gatherTile();
      commitModal();
      toast("Visual reference approved.");
    });
}

function glareLineForm() {
  const line=state.tile.glare_line;
  const regs=state.tile.regions.filter(item=>item.purpose!=="ignore");
  if (regs.length<2) throw Error("Draw at least two sampling regions to mark the two confirmed examples.");
  const parallel=state.tile.photos.filter(item=>item.role==="parallel"&&item.registered_url);
  const cross=state.tile.photos.filter(item=>item.role==="cross"&&item.registered_url);
  const examples=line?.examples||[];
  const glareExample=examples.find(item=>item.kind==="glare");
  const colorExample=examples.find(item=>item.kind==="color");
  const regionChoices=regs.map(item=>option(item.id,item.name)).join("");
  modal(line?"Replace the specimen's glare line":"Adopt the specimen's glare line",`
    <p class="notice">One line per specimen. Mark the patch where the film glare is confirmed and the patch where the bright yarn color is confirmed. Every polarised run and reference re-check reuses this same line.</p>
    <div class="form-columns">${selectField("Parallel-polarized photo","parallel_id",sources(parallel,null,"Choose a capture"))}
    ${selectField("Cross-polarized photo","cross_id",sources(cross,null,"Choose a capture"))}</div>
    <div class="form-columns">${selectField("Confirmed film glare","glare_region",regionChoices)}</div>
    <div class="form-columns">${selectField("Confirmed bright yarn color","color_region",regionChoices)}</div>`,
    line?"Replace line":"Adopt line",async data=>{
      if (data.get("glare_region")===data.get("color_region")) throw Error("Choose two different example regions.");
      await patchTile("/glare-line","PUT",{
        parallel_id:data.get("parallel_id"),cross_id:data.get("cross_id"),
        examples:[{kind:"glare",region_id:data.get("glare_region")},{kind:"color",region_id:data.get("color_region")}],
      },"Glare line adopted.");
    },false,line?btn("Clear line","glare-line-clear","danger"):"");
  const glareSelect=q("#field-glare_region",dialog);
  const colorSelect=q("#field-color_region",dialog);
  if (glareSelect&&glareExample) glareSelect.value=glareExample.region_id;
  if (colorSelect&&colorExample) colorSelect.value=colorExample.region_id;
}

function pairForm() {
  const ready=state.tile.photos.filter(item=>item.registered_url);
  if (ready.length<2) throw Error("Register at least two captures first.");
  modal("Compare registered views",`
    <div class="form-columns">${selectField("First capture","first",sources(ready,ready[0].id))}${selectField("Second capture","second",sources(ready,ready[1].id))}
    ${selectField("View","mode",[["checker","Checkerboard"],["split","Split view"],["edges","Edge disagreement"],["support","Common support"]].map(([value,label])=>option(value,label,"checker")).join(""))}
    ${numField("Divider fraction","divider",0.5,0.05,0.95,0.01)}</div>
    <div id="pair-preview"></div>`,"Compare views",async data=>{
      const token=dialogToken;
      const res=await request(tileUrl()+"/viewer/pair",{method:"POST",body:parseDatesAndNumbers(data,["divider"])});
      const uri=URL.createObjectURL(await res.blob());
      if (token!==dialogToken) {URL.revokeObjectURL(uri);return;}
      const support=JSON.parse(res.headers.get("X-WeaveDesk-Pair"));
      const parent=q("#pair-preview");
      if (parent.dataset.blob) URL.revokeObjectURL(parent.dataset.blob);
      parent.dataset.blob=uri;
      parent.innerHTML=`<figure class="stage">${img(uri,"Registered pair comparison")}</figure>
        <div class="metrics">${stat("Shared pixels",n(support.support_overlap_px,0))}${stat("Edges compared",n(support.edge_positions_compared,0))}</div>`;
    },true);
}

function geometryMeasurementForm(mode) {
  const base=state.tile.photos.find(item=>item.registration_state==="base");
  if (!base) throw Error("Set a base photo before measuring its geometry.");
  const area=mode==="area";
  if (area&&!state.tile.regions.some(item=>item.purpose!=="ignore")) throw Error("Draw an inspection region before measuring its area.");
  const title=area?"Measure region area":"Trace a path";
  const content=area?selectField("Region","region_id",regions(null,false)):`
    <p class="notice">Click along the shape you want to trace. The line reports measured lengths along the registered surface.</p>
    ${drawing(base.registered_url,"Specimen outline for tracing")}
    <div class="actions">${btn("Clear points","draw-clear","tiny secondary")}${btn("Undo point","draw-undo","tiny ghost")}<span class="soft" id="draw-count"></span></div>`;
  modal(title,textField("Measurement title","title",area?"Region area":"Surface trace",'required maxlength="90"')+content,
    "Save geometry",async data=>{
      const payload={title:data.get("title"),mode,region_id:data.get("region_id"),points:drawPoints};
      state.metrologyActive=await api(tileUrl()+"/metrology",{method:"POST",body:payload});
      await gatherTile();
      commitModal();
      toast("Geometry measurement saved.");
      render();
    },!area);
  if (!area) prepareDrawing([],45);
}

function importEvidence() {
  modal("Verify evidence ZIP",field("Evidence ZIP","packet",'<input id="field-packet" name="packet" type="file" accept="application/zip,.zip" required>')+
    '<p class="soft">Select a WeaveDesk inspection archive up to 24 MiB. The viewer checks the declared images against their embedded digests.</p>',
    "Verify archive",async data=>{
      state.receivedActive=await api("/api/received",{method:"POST",body:data});
      state.received=(await api("/api/received")).items;
      commitModal();
      toast("Evidence packet verified.");
      render();
    });
}

function qualityForm() {
  modal("Image quality",`
    <div class="form-columns">${selectField("Photo","photo_id",sources(state.tile.photos.filter(photo=>photo.registered_url),currentPhoto()?.id,"Base capture"))}
    ${numField("Grid across","grid",5,3,10)}
    ${selectField("Map layer","mode",option("edge_clipping","Clipping","edge_clipping")+option("local_detail","Local contrast")+option("source_support","Support"))}
    ${selectField("Region","region_id",regions())}</div><div id="quality-report"></div>`,
    "Inspect capture",async data=>{
      const stamp=dialogToken;
      const body=parseDatesAndNumbers(data,["grid"]);
      const [record,capture]=await Promise.all([
        api(tileUrl()+"/quality",{method:"POST",body}),request(tileUrl()+"/quality/overlay",{method:"POST",body}),
      ]);
      const uri=URL.createObjectURL(await capture.blob());
      if (stamp!==dialogToken) {URL.revokeObjectURL(uri);return;}
      const output=q("#quality-report");
      if (output.dataset.blob) URL.revokeObjectURL(output.dataset.blob);
      output.dataset.blob=uri;
      output.innerHTML=`<figure class="stage">${img(uri,"Capture quality map")}</figure><div class="metrics">${stat("Patches above 2% clipping",record.patches_with_clipping)}${stat("Overall local detail p90",n(record.global_detail?.["90"],5))}</div>
      <p class="notice">${e(record.meaning)}</p><details><summary>Recorded exposure inputs</summary>${table(["Role","Aperture","Exposure s","ISO"],record.captures.map(item=>[item.role,item.f_number,item.exposure_seconds,item.iso]))}</details>${table(["Cell","Support","Clipping","Detail p90"],record.cells.map((entry,index)=>[index+1,percent(entry.supported_percent),percent(entry.clipped_percent),n(entry.detail_quantiles["90"],5)]))}`;
    });
}

function pinForm() {
  if (!state.results.length) throw Error("Run a measurement before pinning evidence.");
  const rows=state.results.map(item=>`<label class="row field check"><input type="checkbox" name="result_id" value="${item.id}"${state.result?.id===item.id?" checked":""}>
    <span class="grow">${e(item.label)} · revision ${item.source_version}</span></label>`).join("");
  modal("Pin inspection evidence",`
    ${textField("Evidence title","title",state.tile.name+" inspection",'required maxlength="115"')}
    <div class="divider-list scroll-list">${rows}</div>${areaField("Notes","notes","",2400)}
    ${textField("Tags, comma separated","tags")}
    ${selectField("Use image layer","view",option("overlay","Measurement maps","overlay")+option("source","Registered source"))}
    ${checkField("Include original uploads in ZIP","include_sources",false)}`,
    "Pin evidence",async data=>{
      await api(tileUrl()+"/evidence",{method:"POST",body:{
        title:data.get("title"),result_ids:data.getAll("result_id"),notes:data.get("notes"),view:data.get("view"),
        tags:String(data.get("tags")).split(",").map(value=>value.trim()).filter(Boolean),include_sources:data.has("include_sources"),
      }});
      await gatherTile();
      state.tab="evidence";
      commitModal();
      toast("Inspection evidence pinned.");
      render();
    });
}

function studyOptions(kind) {
  const metrics=state.config.study_metrics[kind];
  q("#field-metric",dialog).innerHTML=Object.entries(metrics).map(([id,label])=>option(id,label)).join("");
  const relevant=reviewUploads.filter(row=>row.kind===kind);
  q("#study-results",dialog).innerHTML=relevant.map(row=>`<label class="field check row"><input type="checkbox" name="study_result" value="${row.id}"><span>${e(row.tile_name)} · revision ${row.source_version}</span></label>`).join("")||"<p class='soft'>No saved runs of this kind.</p>";
}

async function studyForm() {
  const available=await api("/api/studies/measurements");
  reviewUploads=available.items;
  const types=Object.keys(state.config.study_metrics);
  const kinds=types.map(kind=>option(kind,state.config.tools[kind].label,types[0])).join("");
  if (!reviewUploads.length) throw Error("Save comparable image measurements from at least two specimens first.");
  modal("Build survey comparison",`
    ${textField("Study title","title","Tray measurements",'required maxlength="100"')}
    <div class="form-columns">${selectField("Measurement kind","study_kind",kinds)}
    ${selectField("Group by","group_by",["firing","glaze","body","batch"].map(key=>option(key,({firing:"Imaging visit",glaze:"Mount condition",body:"Fabric group",batch:"Tray"})[key],"firing")).join(""))}</div>
    ${selectField("Metric","metric","")}
    <p class="soft">Choose one run per specimen.</p><div class="divider-list scroll-list" id="study-results"></div>`,
    "Save study",async data=>{
      const result=await api("/api/studies",{method:"POST",body:{
        title:data.get("title"),group_by:data.get("group_by"),metric:data.get("metric"),result_ids:data.getAll("study_result"),
      }});
      state.study=result;
      state.studies=(await api("/api/studies")).items;
      commitModal();
      toast("Survey comparison saved.");
      render();
    });
  studyOptions(q("#field-study_kind",dialog).value);
}

function parsedOptions(data,tool) {
  return Object.fromEntries(Object.entries(tool.fields).map(([key,config])=>{
    if (config.kind==="point") return [key,[Number(data.get(key+"-0")),Number(data.get(key+"-1"))]];
    if (config.kind==="integer"||config.kind==="number") return [key,Number(data.get(key))];
    if (config.kind==="flag") return [key,data.has(key)];
    return [key,data.get(key)||null];
  }));
}

async function runMeasurement(data) {
  const revision=travel;
  const result=await api(tileUrl()+"/results",{method:"POST",body:{
    kind:state.tool,photo_id:data.get("photo_id")||null,comparison_id:data.get("comparison_id")||null,
    region_id:data.get("region_id")||null,apply_neutral:data.has("apply_neutral"),
    options:parsedOptions(data,state.config.tools[state.tool]),
  }});
  if (revision!==travel) return;
  state.result=result;
  state.layer="overlay";
  await gatherTile();
  toast("Measurement saved.");
  render();
}

async function doAction(name,target) {
  const {id,value,mode,marker}=target.dataset;
  const functions={
    training:async ()=>{
      const rows=await api("/api/training",{method:"POST",body:{}});
      if (state.tiles.some(item=>item.synthetic)&&rows.items.length) {
        location.hash="tile:"+rows.items[0].id;
      } else {
        state.tiles=(await api("/api/specimens")).items;
        render();
        toast("Four synthetic training specimens are ready.");
      }
    },
    "tile-open":()=>{location.hash="tile:"+id;},
    "tile-new":()=>tileForm(),
    "tile-edit":()=>tileForm(state.tile),
    "tile-tab":()=>{state.tab=value;render();},
    "photo-add":()=>captureForm(),
    "photo-edit":()=>captureForm(currentPhoto()),
    "photo-view":()=>{state.capture=id;state.viewer="raw";render();},
    "photo-remove":()=>modal("Remove this photo?",'<p>Earlier evidence retains the source. Resetting a base photo also clears its live sampling regions and registrations.</p>'+
       checkField("Reset the live specimen plane if this is the base","reset_plane"),"Remove photo",async data=>{
       await patchTile("/photos/"+currentPhoto().id,"DELETE",{reset_plane:data.has("reset_plane")},"Photo removed.");
    }),
    viewer:()=>{state.viewer=value;render();},
    geometry:()=>geometryForm(currentPhoto()?.registration_state==="base"||!state.tile.plane?"base":"registration"),
    "geometry-base":()=>geometryForm("base"),
    "geometry-preview":async ()=>{
      if (drawPoints.length!==4) throw Error("Mark four corners before previewing alignment.");
      const stamp=dialogToken;
      const result=await request(tileUrl()+"/geometry/"+id+"/preview",{method:"POST",body:{
        mode,corners:drawPoints,respect_scale:formData().has("respect_scale"),
      }});
      const uri=URL.createObjectURL(await result.blob());
      if (stamp!==dialogToken) {URL.revokeObjectURL(uri);return;}
      const preview=q("#geometry-preview");
      if (preview.dataset.blob) URL.revokeObjectURL(preview.dataset.blob);
      preview.dataset.blob=uri;
      preview.innerHTML=`<figure class="stage">${img(uri,"Proposed rectified photo")}</figure><p class="soft">Visible support ${e(result.headers.get("X-Specimen-Support"))}%</p>`;
    },
    compare:pairForm,
    quality:qualityForm,
    "region-add":()=>regionForm(),
    "region-edit":()=>regionForm(state.tile.regions.find(item=>item.id===id)),
    "region-remove":()=>modal("Remove sampling region?",'<p>Saved measurements retain their snapshot of this region.</p>',"Remove region",async ()=>{
      await patchTile("/regions/"+id,"DELETE",{},"Sampling region removed.");
    }),
    "draw-clear":()=>{drawPoints=[];drawUpdate();},
    "draw-undo":()=>{drawPoints.pop();drawUpdate();},
    neutral:neutralForm,
    "neutral-clear":()=>patchTile("/calibration","DELETE",{},"Neutral-card balance cleared."),
    "glare-line":glareLineForm,
    "glare-line-clear":()=>modal("Clear the specimen's glare line?",'<p>Saved measurements keep the line they used. Polarized glare and Reference match will need a new line.</p>',"Clear line",async ()=>{
      await patchTile("/glare-line","DELETE",{},"Glare line cleared.");
    }),
    "reference-add":refForm,
    "reference-retire":()=>{
      const ref=state.refs.find(item=>item.id===id);
      modal("Retire visual reference?",'<p>Earlier measurements keep the approved sample they used.</p>',"Retire reference",async ()=>{
        await api("/api/library/"+id+"/retire",{method:"POST",body:{version:ref.version}});
        state.refs=(await api("/api/library")).items;
        commitModal();
        toast("Visual reference retired.");
        render();
      });
    },
    "mark-profile":lineForm,
    "trace-add":()=>geometryMeasurementForm("trace"),
    "area-add":()=>geometryMeasurementForm("area"),
    "metrology-open":()=>{state.metrologyActive=state.metrology.find(item=>item.id===id);render();},
    "receive-add":importEvidence,
    "received-open":()=>{state.receivedActive=state.received.find(item=>item.id===id);render();},
    "defect-note":()=>{
      const record=state.result;
      const old=record.decisions[marker]||{};
      modal("Review "+marker, selectField("Disposition","decision",["review","confirm","dismiss"].map(item=>option(item,item,old.decision||"review")).join(""))+
        areaField("Observation note","notes",old.notes,240),"Save observation",async data=>{
        state.result=await api(`/api/results/${record.id}/observations/${marker}`,{method:"PUT",body:{version:record.version,...plainData(data)}});
        commitModal();
        toast("Observation reviewed.");
        render();
      });
    },
    "result-open":async ()=>{
      state.result=await api("/api/results/"+id);
      state.layer="overlay";
      state.tab="measure";
      render();
    },
    "result-layer":()=>{state.layer=value;render();},
    "defect-choice":async ()=>{
      state.result=await api(`/api/results/${state.result.id}/observations/${marker}`,{method:"PUT",body:{version:state.result.version,decision:value,notes:""}});
      toast("Observation reviewed.");
      render();
    },
    pin:pinForm,
    "study-add":studyForm,
    "study-open":()=>{state.study=state.studies.find(item=>item.id===id);render();},
    close:commitModal,
  };
  if (functions[name]) await functions[name]();
}

function displayFault(err, inside=false) {
  if (inside && dialog.open) {
    q(".dialog-error",dialog).textContent=err.message;
  } else {
    toast(err.message,true);
  }
}

async function runAction(target) {
  if (target.disabled) return;
  target.disabled=true;
  try {
    await doAction(target.dataset.do,target);
  } catch (err) {
    displayFault(err,dialog.contains(target));
  } finally {
    target.disabled=false;
  }
}

document.addEventListener("click",event=>{
  const target=event.target.closest("[data-do]");
  if (target) {void runAction(target);return;}
  const drawing=event.target.closest("#draw-stage");
  if (drawing && drawPoints.length<drawLimit) {
    const rect=drawing.getBoundingClientRect();
    const u=Math.min(1,Math.max(0,(event.clientX-rect.left)/rect.width));
    const v=Math.min(1,Math.max(0,(event.clientY-rect.top)/rect.height));
    drawPoints.push([Number(u.toFixed(6)),Number(v.toFixed(6))]);
    drawUpdate();
  }
});

document.addEventListener("submit",async event=>{
  const form=event.target;
  if (!["measurement-form","dialog-form"].includes(form.id)) return;
  event.preventDefault();
  if (working) return;
  working++;
  const button=form.querySelector('[type="submit"]');
  const error=form.querySelector(".dialog-error");
  const data=new FormData(form);
  if (button) button.disabled=true;
  if (error) error.textContent="";
  try {
    if (form.id==="measurement-form") await runMeasurement(data);
    else if (currentForm) await currentForm(data);
  } catch(err) {
    displayFault(err,form.id==="dialog-form");
  } finally {
    working--;
    if (button) button.disabled=false;
  }
});

document.addEventListener("input",event=>{
  if (event.target.id==="tile-search") {
    state.query=event.target.value;
    q("#tile-list").innerHTML=cardTiles();
  }
});

document.addEventListener("change",event=>{
  if (event.target.id==="field-kind") {
    state.tool=event.target.value;
    const existing=q("#measurement-form");
    existing.outerHTML=measureControls();
    q("#field-kind")?.focus();
  }
  if (event.target.id==="field-study_kind") studyOptions(event.target.value);
});

dialog.addEventListener("close",()=>{
  qa("[data-blob]",dialog).forEach(el=>URL.revokeObjectURL(el.dataset.blob));
});
window.addEventListener("hashchange",()=>{void route().catch(displayFault);});
api("/api/configuration").then(config=>{state.config=config;return route();}).catch(displayFault);
