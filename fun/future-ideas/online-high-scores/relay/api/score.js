// Score relay: a tiny serverless function (Vercel) between the games and GitHub.
// The browser can't hold a GitHub token, so the page POSTs the score here; this function checks it
// and asks GitHub to run .github/workflows/scores.yml, which validates again and commits the score.
// Needs one environment variable in Vercel: GITHUB_TOKEN = fine-grained token with
// "Contents: Read and write" on katherinekarow/portfolio only. The token never reaches the browser.
const REPO = "katherinekarow/portfolio";
const ALLOWED = ["https://katherinekarow.github.io"];
const LEVELS = { sudoku: ["easy", "medium", "hard", "expert"], minesweeper: ["beginner", "intermediate", "expert"] };
const hits = new Map();                         // best-effort rate limit per IP (resets when the function cold-starts)

export default async function handler(req, res) {
  const origin = req.headers.origin || "";
  if (ALLOWED.includes(origin)) res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  if (!ALLOWED.includes(origin)) return res.status(403).json({ error: "Scores can only be sent from the portfolio site." });

  const ip = (req.headers["x-forwarded-for"] || "").split(",")[0].trim() || "unknown";
  const now = Date.now(), recent = (hits.get(ip) || []).filter(t => now - t < 10 * 60 * 1000);
  if (recent.length >= 5) return res.status(429).json({ error: "Too many scores from here. Try again in a few minutes." });
  hits.set(ip, [...recent, now]);

  const b = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
  const name = String(b.name || "").trim();
  const seconds = Number(b.seconds);
  if (!LEVELS[b.game] || !LEVELS[b.game].includes(b.level)) return res.status(400).json({ error: "Unknown game or level." });
  if (!/^[A-Za-z0-9 _.\-]{1,16}$/.test(name)) return res.status(400).json({ error: "Name: 1-16 letters, numbers, spaces or _ . -" });
  if (!Number.isFinite(seconds) || seconds <= 0 || seconds > 4 * 3600) return res.status(400).json({ error: "Invalid time." });

  const r = await fetch(`https://api.github.com/repos/${REPO}/dispatches`, {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.GITHUB_TOKEN}`, Accept: "application/vnd.github+json", "User-Agent": "score-relay" },
    body: JSON.stringify({ event_type: "score", client_payload: { game: b.game, level: b.level, name, seconds } }),
  });
  if (r.status !== 204) return res.status(502).json({ error: "GitHub didn't accept the score. Please try again later." });
  return res.status(202).json({ ok: true });
}
