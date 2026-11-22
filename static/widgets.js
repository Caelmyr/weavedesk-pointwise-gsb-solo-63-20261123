const icons = {
  logo: "<path d='M4 6l7.8-2 8 3-2 11-8 2L4 16Z'/><path d='M4 6l7 4 8.8-3M11 10l-1.2 10M12 7l3-1'/>",
  tiles: "<rect x='3' y='3' width='7' height='7' rx='1'/><rect x='14' y='3' width='7' height='7' rx='1'/><rect x='3' y='14' width='7' height='7' rx='1'/><rect x='14' y='14' width='7' height='7' rx='1'/>",
  book: "<path d='M5 4h14v15H8a3 3 0 0 1-3-3V4Zm0 12a2 2 0 0 1 2-2h12M9 8h6'/>",
  plot: "<path d='M3 4v16h18M7 14l4-4 4 2 5-6'/>",
  archive: "<rect x='3' y='4' width='18' height='5' rx='1'/><path d='M5 9v11h14V9m-9 4h4'/>",
  plus: "<path d='M12 5v14M5 12h14'/>",
  arrow: "<path d='M5 12h14m-6-6 6 6-6 6'/>",
  back: "<path d='M19 12H5m6-6-6 6 6 6'/>",
  search: "<circle cx='10.7' cy='10.8' r='5.6'/><path d='m15 15 5 5'/>",
  camera: "<path d='M5 7h3l1.5-2h5L16 7h3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2Z'/><circle cx='12' cy='13' r='3'/>",
  frame: "<path d='M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M8 12h8m-4-4v8'/>",
  measure: "<path d='M3 16l13-13 5 5L8 21l-5-5zm4-4 2 2m1-5 2 2m1-5 2 2'/>",
  check: "<path d='m5 12 4 4L19 6'/>",
  x: "<path d='m6 6 12 12M18 6 6 18'/>",
  download: "<path d='M12 3v12m-5-5 5 5 5-5M5 17v3h14v-3'/>",
  sample: "<path d='m7 3 10 2 4 11-7 5-11-3 4-15Z'/><circle cx='13' cy='12' r='2'/>",
  clock: "<circle cx='12' cy='12' r='8'/><path d='M12 8v5l3 2'/>",
  edit: "<path d='m5 16 11-11 3 3L8 19H5v-3zm9-9 3 3'/>",
  compare: "<rect x='3' y='5' width='18' height='14' rx='2'/><path d='M12 5v14m-4-7h1m6 0h1'/>",
};

export const q = (selector, root = document) => root.querySelector(selector);
export const qa = (selector, root = document) => Array.from(root.querySelectorAll(selector));
export const icon = name => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${icons[name] || ""}</svg>`;
export const e = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
export const n = (value, digits = 2) => value == null ? "—" : Number(value).toLocaleString("en", {maximumFractionDigits:digits});
export const date = value => value ? new Date(value).toLocaleDateString(undefined,{month:"short",day:"numeric",year:"numeric"}) : "—";
export const percent = value => value == null ? "—" : `${n(value)}%`;
export const option = (value,label,selected) => `<option value="${e(value)}"${String(value) === String(selected) ? " selected" : ""}>${e(label)}</option>`;
export const btn = (label, action, style = "secondary", glyph, data = "") =>
  `<button class="button ${style}" type="button" data-do="${action}" ${data}>${glyph ? icon(glyph) : ""}${e(label)}</button>`;
export const link = (label, href, style = "secondary", glyph) =>
  `<a class="button ${style}" href="${e(href)}">${glyph ? icon(glyph) : ""}${e(label)}</a>`;
export const badge = (label, style = "") => `<span class="badge ${style}">${e(label)}</span>`;
export const empty = (message, action = "") => `<div class="empty"><p>${e(message)}</p>${action}</div>`;
export const field = (label,name,body) => `<div class="field"><label for="field-${name}">${e(label)}</label>${body}</div>`;
export const textField = (label,name,value="",attrs="") => field(label,name,`<input id="field-${name}" name="${name}" value="${e(value)}" ${attrs}>`);
export const numField = (label,name,value,min,max,step=1) => textField(label,name,value,`type="number" min="${min}" max="${max}" step="${step}"`);
export const selectField = (label,name,choices) => field(label,name,`<select id="field-${name}" name="${name}">${choices}</select>`);
export const areaField = (label,name,value="",limit=2000) => field(label,name,`<textarea id="field-${name}" name="${name}" maxlength="${limit}">${e(value)}</textarea>`);
export const checkField = (label,name,checked=false) => `<label class="field check"><input type="checkbox" name="${name}"${checked ? " checked" : ""}> ${e(label)}</label>`;
export const img = (url,label,extras="") => `<img src="${e(url)}" alt="${e(label)}" ${extras}>`;
export const stat = (label,value) => `<div class="stat"><strong>${e(value)}</strong><span>${e(label)}</span></div>`;
export const dots = (name,color) => `<span><i class="dot" style="background:${e(color)}"></i>${e(name)}</span>`;

export async function request(url, settings = {}) {
  const {body, ...options} = settings;
  const binary = body instanceof FormData;
  const res = await fetch(url, {
    ...options,
    ...(body == null ? {} : {body:binary ? body : JSON.stringify(body)}),
    headers: {...options.headers, ...(body != null && !binary ? {"Content-Type":"application/json"} : {})},
  });
  if (!res.ok) {
    const info = await res.json().catch(() => ({}));
    throw Error(info.error || `Request failed (${res.status}).`);
  }
  return res;
}

export async function api(url, settings) {
  return (await request(url,settings)).json();
}

export function plot(series, label, options = {}) {
  const lines = series.filter(item => item.values.length);
  const all = lines.flatMap(item=>item.values).filter(value=>value != null && Number.isFinite(value));
  if (!all.length) return '<p class="plot-note">There are no supported positions to plot.</p>';
  const lo = options.min ?? Math.min(...all), hi = options.max ?? Math.max(...all);
  const height = 147, width = 530, left = 42, top = 15, bottom = height - 26, right = width - 16;
  const point = (value,index,count) => [
    left + index * (right-left) / Math.max(1,count-1),
    bottom - (value-lo) / Math.max(hi-lo,.00001) * (bottom-top),
  ];
  const paths = lines.map(item => {
    let drawing = false;
    const coords = item.values.map((value,index)=>{
      if (value == null || !Number.isFinite(value)) {drawing=false;return "";}
      const [x,y]=point(value,index,item.values.length);
      const command = drawing ? "L" : "M";
      drawing=true;
      return `${command}${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    return `<path d="${coords}" fill="none" stroke="${e(item.color || "#377360")}" stroke-width="2"/>`;
  }).join("");
  const axes = [lo,(lo+hi)/2,hi].map(value=>{
    const [,y]=point(value,0,1);
    return `<path d="M${left} ${y} H${right}" stroke="#dadfd1"/><text x="${left-5}" y="${y+4}" text-anchor="end" fill="#5d6d60" font-size="10">${e(n(value,3))}</text>`;
  }).join("");
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${e(label)}">${axes}${paths}<text x="${(left+right)/2}" y="${height-6}" text-anchor="middle" fill="#5d6d60" font-size="10">${e(options.axis || label)}</text></svg>`;
}

export function table(columns,rows) {
  return `<div class="table-wrap"><table><thead><tr>${columns.map(value=>`<th>${e(value)}</th>`).join("")}</tr></thead><tbody>
    ${rows.map(row=>`<tr>${row.map(value=>`<td>${e(value ?? "—")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
