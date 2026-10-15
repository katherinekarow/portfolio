// Small shared toolkit for the gallery: formatting, a tooltip, and three SVG chart types
// (line with crosshair, horizontal bars with optional interval, heat table helpers).
// No libraries - every dashboard is a static page reading its own data.json.
const D = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const SERIES = ["--s1", "--s2", "--s3", "--s4"];
  const color = i => css(SERIES[i % SERIES.length]);

  const fmt = {
    int: v => v == null ? "—" : Math.round(v).toLocaleString("en-US"),
    usd: v => v == null ? "—" : "$" + Math.round(v).toLocaleString("en-US"),
    usd2: v => v == null ? "—" : "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
    usdk: v => v == null ? "—" : Math.abs(v) >= 1e6 ? "$" + (v / 1e6).toFixed(2) + "M" : "$" + (v / 1e3).toFixed(0) + "k",
    pct: (v, d = 1) => v == null ? "—" : (v * 100).toFixed(d) + "%",
    pp: (v, d = 1) => v == null ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + " pts",
    delta: (v, d = 1) => v == null ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%",
    num: (v, d = 2) => v == null ? "—" : Number(v).toFixed(d),
  };

  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function h(tag, attrs = {}, html) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (html != null) e.innerHTML = html;
    return e;
  }
  function niceTicks(lo, hi, n = 5) {
    if (hi === lo) hi = lo + 1;
    const step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
    const t = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
    return t;
  }
  function tip(wrap) {
    let t = wrap.querySelector(".tip");
    if (!t) { t = h("div", { class: "tip" }); wrap.appendChild(t); }
    return {
      show(html, x, y) {
        t.innerHTML = html; t.style.display = "block";
        const W = wrap.clientWidth, tw = t.offsetWidth;
        t.style.left = Math.min(Math.max(x + 14, 0), W - tw - 4) + "px";
        t.style.top = Math.max(y - 10, 0) + "px";
      },
      hide() { t.style.display = "none"; },
    };
  }
  const row = (label, val, sw) => `<div><span>${sw ? `<span class="sw" style="background:${sw}"></span> ` : ""}${label}</span><span>${val}</span></div>`;

  // Charts are drawn at the container's pixel width so text stays 11px; redrawn on resize.
  const width = svg => Math.max(300, Math.round(svg.parentElement.clientWidth || 760));
  const drawn = new Map();
  const remember = (svg, fn, opts) => drawn.set(svg, [fn, opts]);
  let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => drawn.forEach(([fn, o], svg) => svg.isConnected && fn(svg, o)), 150); });

  // Line chart. series: [{name, values:[...], color, dash, band:[lo[],hi[]] }]; x: labels array.
  function line(svg, { x, series, yFmt = fmt.int, xFmt = v => v, yMin, yMax, title = "", xTicks = 6, refs = [] }) {
    svg.innerHTML = ""; remember(svg, line, arguments[1]);
    const W = width(svg), H = W < 600 ? 240 : 300, m = { l: 56, r: 16, t: 14, b: 30 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const all = series.flatMap(s => [...s.values, ...(s.band ? [...s.band[0], ...s.band[1]] : [])]).filter(v => v != null);
    let lo = yMin ?? Math.min(...all), hi = yMax ?? Math.max(...all);
    const pad = (hi - lo) * 0.06; if (yMin == null) lo -= pad; if (yMax == null) hi += pad;
    const ticks = niceTicks(lo, hi); lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
    const n = x.length, X = i => m.l + (W - m.l - m.r) * (n === 1 ? 0.5 : i / (n - 1)), Y = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
    ticks.forEach(t => { el("line", { class: "gl", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }, svg); const tx = el("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, svg); tx.textContent = yFmt(t); });
    const step = Math.max(1, Math.round(n / xTicks));
    for (let i = 0; i < n; i += step) { const tx = el("text", { x: X(i), y: H - 8, "text-anchor": "middle" }, svg); tx.textContent = xFmt(x[i]); }
    refs.forEach(r => { el("line", { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), stroke: css("--muted"), "stroke-dasharray": "4 4", "stroke-width": 1 }, svg); const t = el("text", { x: W - m.r, y: Y(r.y) - 4, "text-anchor": "end" }, svg); t.textContent = r.label; });
    const path = vals => { let d = "", pen = false; vals.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + X(i).toFixed(1) + "," + Y(v).toFixed(1); pen = true; }); return d; };
    series.forEach(s => {
      if (s.band) {
        const [a, b] = s.band; let d = "";
        a.forEach((v, i) => d += (i ? "L" : "M") + X(i) + "," + Y(v));
        for (let i = b.length - 1; i >= 0; i--) d += "L" + X(i) + "," + Y(b[i]);
        el("path", { d: d + "Z", fill: s.bandColor || css("--band"), stroke: "none" }, svg);
      }
    });
    series.forEach(s => el("path", { d: path(s.values), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round", "stroke-linecap": "round" }, svg));
    // hover layer
    const wrap = svg.parentElement, tp = tip(wrap);
    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: css("--muted"), "stroke-width": 1, visibility: "hidden" }, svg);
    const dots = series.map(s => el("circle", { r: 4, fill: s.color, stroke: css("--panel"), "stroke-width": 2, visibility: "hidden" }, svg));
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    const move = ev => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
      const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / (W - m.l - m.r) * (n - 1))));
      cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
      let html = `<b>${title}${xFmt(x[i])}</b>`;
      series.forEach((s, k) => { const v = s.values[i]; if (v == null) { dots[k].setAttribute("visibility", "hidden"); return; } dots[k].setAttribute("cx", X(i)); dots[k].setAttribute("cy", Y(v)); dots[k].setAttribute("visibility", "visible"); html += row(s.name, yFmt(v), s.color); });
      const wr = wrap.getBoundingClientRect();
      tp.show(html, ev.clientX - wr.left, ev.clientY - wr.top);
    };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", () => { tp.hide(); cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); });
  }

  // Horizontal bars. items: [{label, value, lo, hi, color, tip}]; zero line always shown.
  function hbars(svg, { items, xFmt = fmt.int, ref, refLabel = "", xMin, xMax }) {
    svg.innerHTML = ""; remember(svg, hbars, arguments[1]);
    const W = width(svg), rowH = 30, m = { l: Math.min(170, Math.max(...items.map(d => String(d.label).length)) * 6.6 + 14), r: 60, t: 8, b: 26 }, H = m.t + m.b + rowH * items.length;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const vals = items.flatMap(d => [d.value, d.lo ?? d.value, d.hi ?? d.value, 0]);
    let lo = xMin ?? Math.min(...vals), hi = xMax ?? Math.max(...vals);
    const ticks = niceTicks(lo, hi); lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
    const X = v => m.l + (W - m.l - m.r) * (v - lo) / (hi - lo);
    ticks.forEach(t => { el("line", { class: "gl", x1: X(t), x2: X(t), y1: m.t, y2: H - m.b }, svg); const tx = el("text", { x: X(t), y: H - 8, "text-anchor": "middle" }, svg); tx.textContent = xFmt(t); });
    el("line", { class: "ax", x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: css("--muted") }, svg);
    if (ref != null) { el("line", { x1: X(ref), x2: X(ref), y1: m.t, y2: H - m.b, stroke: css("--muted"), "stroke-dasharray": "4 4" }, svg); const t = el("text", { x: X(ref) + 4, y: m.t + 10 }, svg); t.textContent = refLabel; }
    const wrap = svg.parentElement, tp = tip(wrap);
    items.forEach((d, i) => {
      const y = m.t + i * rowH + 6, bh = rowH - 12;
      const lab = el("text", { x: m.l - 8, y: y + bh / 2 + 4, "text-anchor": "end" }, svg); lab.textContent = d.label; lab.style.fill = css("--ink");
      const x0 = X(Math.min(0, d.value)), x1 = X(Math.max(0, d.value));
      const bar = el("rect", { x: x0, y, width: Math.max(1, x1 - x0), height: bh, rx: 4, fill: d.color }, svg);
      if (d.lo != null) {
        el("line", { x1: X(d.lo), x2: X(d.hi), y1: y + bh / 2, y2: y + bh / 2, stroke: css("--ink"), "stroke-width": 1.5 }, svg);
        [d.lo, d.hi].forEach(v => el("line", { x1: X(v), x2: X(v), y1: y + 3, y2: y + bh - 3, stroke: css("--ink"), "stroke-width": 1.5 }, svg));
      }
      const vt = el("text", { x: X(Math.max(d.value, d.hi ?? d.value)) + 6, y: y + bh / 2 + 4 }, svg); vt.textContent = xFmt(d.value);
      const hitr = el("rect", { x: m.l, y: y - 6, width: W - m.l - m.r, height: rowH, fill: "transparent" }, svg);
      hitr.addEventListener("pointermove", ev => { const wr = wrap.getBoundingClientRect(); bar.setAttribute("opacity", .8); tp.show(d.tip || `<b>${d.label}</b>${row("Value", xFmt(d.value))}`, ev.clientX - wr.left, ev.clientY - wr.top); });
      hitr.addEventListener("pointerleave", () => { bar.setAttribute("opacity", 1); tp.hide(); });
    });
  }

  // Sortable table. cols: [{k, label, fmt, cls, html(row)}]
  function table(container, { cols, rows, sortKey, desc = true, onRow }) {
    let key = sortKey, dsc = desc;
    const draw = () => {
      const rs = [...rows].sort((a, b) => { const x = a[key], y = b[key]; const c = (x > y) - (x < y); return dsc ? -c : c; });
      const head = cols.map(c => `<th class="${c.cls || ""}" data-k="${c.k}" aria-sort="${c.k === key ? (dsc ? "descending" : "ascending") : "none"}">${c.label}${c.k === key ? (dsc ? " ↓" : " ↑") : ""}</th>`).join("");
      const body = rs.map((r, i) => `<tr data-i="${rows.indexOf(r)}">` + cols.map(c => `<td class="${c.cls || ""}">${c.html ? c.html(r) : (c.fmt || (v => v))(r[c.k])}</td>`).join("") + "</tr>").join("");
      container.innerHTML = `<div class="tablewrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
      container.querySelectorAll("th[data-k]").forEach(th => th.onclick = () => { const k = th.dataset.k; if (k === key) dsc = !dsc; else { key = k; dsc = true; } draw(); });
      if (onRow) container.querySelectorAll("tbody tr").forEach(tr => { tr.style.cursor = "pointer"; tr.onclick = () => onRow(rows[+tr.dataset.i]); });
    };
    draw();
    return { set(r) { rows = r; draw(); } };
  }

  // Diverging fill for a value in [-span, span]: blue for above, orange for below, neutral grey at 0.
  function diverge(v, span) {
    const t = Math.max(-1, Math.min(1, v / span)), a = Math.abs(t);
    const c = t >= 0 ? css("--div-pos") : css("--div-neg");
    return `color-mix(in srgb, ${c} ${Math.round(a * 70)}%, var(--grid))`;
  }

  const load = (path = "data.json") => fetch(path).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
  const kpis = (container, items) => container.innerHTML = items.map(k => `<div class="kpi"><div class="v">${k.v}</div><div class="l">${k.l}</div>${k.d ? `<div class="d">${k.d}</div>` : ""}</div>`).join("");

  return { fmt, el, h, css, color, line, hbars, table, tip, row, diverge, load, kpis, niceTicks };
})();
