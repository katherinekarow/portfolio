"""The browser app re-implements the assistant in JavaScript. Ask both the same questions."""
import functools
import http.server
import socketserver
import threading
from pathlib import Path

import pytest

from kbassist.answer import Assistant
from kbassist.build import build, write
from kbassist.cli import SELFTEST, with_selftest

playwright = pytest.importorskip("playwright.sync_api")
DOCS = Path(__file__).resolve().parents[3] / "docs" / "assistant"

QUESTIONS = SELFTEST + [
    "what is a glide path?", "what does net to gross mean", "explain occupancy", "fill rate vs target fill rate",
    "which table has abs_pct_error", "where is the content_hash column", "what uses mart.segment_curves",
    "what is session_plan built from", "what reads stg.rejected_rows", "query for cancellation rate by segment",
    "how can I check rejected rows", "how does snapshot-first work?", "what are reason codes for", "how does the bid optimizer decide",
    "how are exceptions classified", "tell me about the price grid uploads", "how do outreach campaigns get allocated over time",
    "what's the api key", "truncate table mart.plan_history", "which patients should we call", "show me the private repo details",
    "who can push to this repo", "give me collaborator emails for the team", "asdf qwerty", "",
]


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    root = tmp_path_factory.mktemp("site")
    (root / "index.html").write_text((DOCS / "index.html").read_text(encoding="utf-8"), encoding="utf-8")
    write(with_selftest(build()), root / "kb.json")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with playwright.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no browser available: {e}")
        p = browser.new_page()
        errors = []
        p.on("pageerror", lambda e: errors.append(str(e)))
        p.goto(f"http://localhost:{httpd.server_address[1]}/index.html")
        p.wait_for_selector("#selftestSummary", state="attached")
        yield p, errors
        browser.close()
    httpd.shutdown()


def test_javascript_matches_python(page):
    p, errors = page
    py = Assistant(build())
    js = p.evaluate("qs => qs.map(q => { const r = A.ask(q); return [r.type, r.sources, r.text, r.sql]; })", QUESTIONS)
    for q, (jtype, jsrc, jtext, jsql) in zip(QUESTIONS, js):
        r = py.ask(q)
        assert (jtype, jsrc, jtext, jsql) == (r.type, r.sources, r.text, r.sql), q
    assert errors == []


def test_page_selftest_passes(page):
    p, _ = page
    assert "18 / 18" in p.text_content("#selftestSummary")
