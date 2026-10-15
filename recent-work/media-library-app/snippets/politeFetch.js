/**
 * politeFetch — call a public API without hammering it.
 *  - waits at least `minGapMs` between requests to the same host
 *  - retries on 429/5xx with exponential backoff (honours Retry-After)
 *  - always sends an honest User-Agent, which many public APIs require
 *
 * Rewritten example of the approach used in a private project.
 */
const lastCall = new Map();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function politeFetch(url, { minGapMs = 250, retries = 3, userAgent = "CatalogApp/1.0", ...init } = {}) {
  const host = new URL(url).host;
  for (let attempt = 0; ; attempt++) {
    const wait = (lastCall.get(host) ?? 0) + minGapMs - Date.now();
    if (wait > 0) await sleep(wait);
    lastCall.set(host, Date.now());

    const res = await fetch(url, { ...init, headers: { "User-Agent": userAgent, Accept: "application/json", ...init.headers } });
    const retryable = res.status === 429 || res.status >= 500;
    if (!retryable || attempt >= retries) return res;

    const retryAfter = Number(res.headers.get("Retry-After"));
    await sleep(Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : 500 * 2 ** attempt);
  }
}

// Usage:
// const res = await politeFetch(`https://openlibrary.org/isbn/${isbn}.json`);
// if (res.ok) console.log(await res.json());
