// Sakura OS shell: menu bar, falling petals, clock and the blossom icon. Shared by every page.
(() => {
  const ROOT = document.currentScript.src.replace(/assets\/sakura\.js.*$/, "");
  const BLOSSOM = `<svg class="blossom" viewBox="0 0 100 100" aria-hidden="true"><g transform="translate(50 50)">${
    [0, 72, 144, 216, 288].map(r => `<path transform="rotate(${r})" d="M0 -6 C -14 -14 -18 -32 -9 -42 L -3 -37 L 0 -44 L 3 -37 L 9 -42 C 18 -32 14 -14 0 -6 Z" fill="#f6a9c4" stroke="#d6457a" stroke-width="3" stroke-linejoin="round"/>`).join("")
  }<circle r="8" fill="#fff4f8" stroke="#d6457a" stroke-width="3"/><circle r="3" fill="#7b5ea7"/></g></svg>`;
  window.SAKURA = { ROOT, BLOSSOM };

  const page = document.body.dataset.page || "";
  const links = [["", "Home", "home"], ["dashboards/", "Dashboards", "dashboards"], ["formulas/", "Formulas", "formulas"],
    ["assistant/", "Assistant", "assistant"], ["sudoku/", "Sudoku", "sudoku"], ["minesweeper/", "Minesweeper", "minesweeper"]];
  const bar = document.createElement("header");
  bar.className = "menubar";
  bar.innerHTML = `<a class="brand" href="${ROOT}">${BLOSSOM}<span>sakura.os</span></a>
    <nav aria-label="Site">${links.map(([h, t, k]) => `<a href="${ROOT}${h}"${k === page ? ' aria-current="page"' : ""}>${t}</a>`).join("")}</nav>
    <span class="spacer"></span>
    <a href="https://github.com/katherinekarow/portfolio" style="color:inherit;text-decoration:none">GitHub</a>
    <span class="clock" aria-hidden="true"></span>`;
  document.body.prepend(bar);
  const clock = bar.querySelector(".clock");
  const tick = () => clock.textContent = new Date().toLocaleString("en-US", { weekday: "short", hour: "numeric", minute: "2-digit" });
  tick(); setInterval(tick, 30000);

  // replace any <i class="bl"></i> placeholder with the blossom icon
  document.querySelectorAll("i.bl").forEach(i => i.outerHTML = BLOSSOM);

  if (!matchMedia("(prefers-reduced-motion: reduce)").matches && !document.body.hasAttribute("data-no-petals")) {
    const box = document.createElement("div");
    box.className = "petals"; box.setAttribute("aria-hidden", "true");
    const n = innerWidth < 600 ? 7 : 14;
    for (let i = 0; i < n; i++) {
      const s = document.createElement("span");
      const size = 8 + Math.random() * 9;
      Object.assign(s.style, { left: Math.random() * 100 + "vw", width: size + "px", height: size * .8 + "px",
        animationDuration: 11 + Math.random() * 12 + "s", animationDelay: -Math.random() * 20 + "s", opacity: .45 + Math.random() * .4 });
      box.appendChild(s);
    }
    document.body.appendChild(box);
  }
})();
