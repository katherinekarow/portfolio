// Best times for the games, kept on this device (localStorage).
// Online, committed high scores are a planned future project: see fun/future-ideas/online-high-scores.
(() => {
  const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const key = (g, l) => `best:${g}:${l}`;
  const get = (g, l) => { try { return JSON.parse(localStorage.getItem(key(g, l))) || []; } catch { return []; } };
  const put = (g, l, v) => { try { localStorage.setItem(key(g, l), JSON.stringify(v)); } catch {} };
  function record(game, level, seconds) {
    const list = get(game, level);
    const row = { seconds: Math.round(seconds * 10) / 10, date: new Date().toISOString().slice(0, 10) };
    list.push(row); list.sort((a, b) => a.seconds - b.seconds); list.splice(5);
    put(game, level, list);
    return list.indexOf(row) + 1;          // 0 = not in the top 5
  }
  function table(el, game, level) {
    const rows = get(game, level);
    el.innerHTML = rows.length
      ? `<table class="scores"><thead><tr><th>#</th><th>Time</th><th class="l">Date</th></tr></thead><tbody>${
          rows.map((r, i) => `<tr><td>${i + 1}</td><td>${fmt(r.seconds)}</td><td class="l">${r.date}</td></tr>`).join("")}</tbody></table>
         <p class="muted small">Saved in this browser only.</p>`
      : `<p class="muted">No ${level} wins on this device yet. Your best five times will show here.</p>`;
  }
  function offer(el, { game, level, seconds, eligible = true, reason = "" }) {
    if (!eligible) { el.innerHTML = `<p class="muted">${reason}</p>`; return; }
    const rank = record(game, level, seconds);
    el.innerHTML = rank === 1 ? `<p><b>New personal best!</b></p>` : rank ? `<p>That's your #${rank} best ${level} time on this device.</p>` : "";
  }
  window.Scores = { table, offer, record, fmt };
})();
