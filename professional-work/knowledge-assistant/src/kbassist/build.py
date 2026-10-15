"""Build and validate the knowledge base.

Sources
  kb/catalog.yaml      warehouse views: grain, columns, upstream, sample queries
  kb/glossary.yaml     business terms, aliases, formulas
  kb/guardrails.yaml   what to refuse or redirect
  ../*/README.md       project write-ups, split into sections (so the assistant can answer "how does X work?")

Validation fails the build (CatalogError) on anything a user would trip over:
broken lineage, cycles, a layer reading from a later layer, undocumented columns,
names that break the convention, sample queries that reference unknown views.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
import yaml

ROOT = Path(__file__).resolve().parents[2]                 # knowledge-assistant/
LAYERS = ["raw", "stg", "core", "mart", "rpt"]
NAME_RE = re.compile(r"^(raw|stg|core|mart|rpt)\.[a-z][a-z0-9_]*$")


class CatalogError(ValueError):
    pass


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "kb" / name).read_text(encoding="utf-8"))


def validate_catalog(views: list[dict]) -> nx.DiGraph:
    errors = []
    names = [v["name"] for v in views]
    for n in {n for n in names if names.count(n) > 1}:
        errors.append(f"duplicate view {n}")
    known = set(names)
    g = nx.DiGraph()
    for v in views:
        n = v["name"]
        g.add_node(n, layer=v["layer"])
        if not NAME_RE.match(n):
            errors.append(f"{n}: name must be <layer>.<snake_case>")
        elif n.split(".")[0] != v["layer"]:
            errors.append(f"{n}: schema prefix does not match layer {v['layer']}")
        for key in ("grain", "description", "owner", "columns"):
            if not v.get(key):
                errors.append(f"{n}: missing {key}")
        for col, desc in (v.get("columns") or {}).items():
            if not re.match(r"^[a-z][a-z0-9_]*$", col):
                errors.append(f"{n}.{col}: column names must be snake_case")
            if not desc or not str(desc).strip():
                errors.append(f"{n}.{col}: column has no description")
        if v["layer"] == "raw" and v.get("upstream"):
            errors.append(f"{n}: raw views cannot have upstream sources")
        for up in v.get("upstream") or []:
            if up not in known:
                errors.append(f"{n}: upstream {up} does not exist")
                continue
            g.add_edge(up, n)
            if LAYERS.index(up.split(".")[0]) > LAYERS.index(v["layer"]):
                errors.append(f"{n}: reads from later layer {up}")
        for ref in re.findall(r"\b(?:from|join)\s+([a-z_]+\.[a-z0-9_]+)", (v.get("sample_query") or "").lower()):
            if ref not in known:
                errors.append(f"{n}: sample query references unknown view {ref}")
    for cyc in nx.simple_cycles(g):
        errors.append("lineage cycle: " + " -> ".join(cyc))
    if errors:
        raise CatalogError("catalog validation failed:\n  " + "\n  ".join(errors))
    return g


def _strip_md(text: str) -> str:
    text = re.sub(r"```mermaid.*?```", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)                # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)           # links -> text
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[*`>#|]", " ", text)
    text = re.sub(r"-{3,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_body(md: str, limit: int = 700) -> list[str]:
    """Paragraph / table-row aware chunks of at most ~limit characters (after markdown stripping)."""
    blocks, cur = [], ""
    for para in re.split(r"\n\s*\n|\n(?=\|)", md):
        p = _strip_md(para)
        if not p:
            continue
        if cur and len(cur) + len(p) + 1 > limit:
            blocks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        blocks.append(cur)
    return blocks


def readme_chunks(folder: Path, base_url: str) -> list[dict]:
    """Split each project README into ## sections, and long sections into ~700-character passages."""
    chunks = []
    for readme in sorted(folder.glob("*/README.md")):
        project = readme.parent.name
        text = readme.read_text(encoding="utf-8")
        title = re.search(r"^#\s+(.+)$", text, re.M).group(1).strip()
        parts = re.split(r"^##\s+", text, flags=re.M)
        sections = [("Overview", parts[0].split("\n", 1)[1] if "\n" in parts[0] else "")]
        sections += [(p.split("\n", 1)[0].strip(), p.split("\n", 1)[1] if "\n" in p else "") for p in parts[1:]]
        for heading, body in sections:
            anchor = re.sub(r"[^a-z0-9 -]", "", heading.lower()).strip().replace(" ", "-")
            body = re.sub(r"```mermaid.*?```", "", body, flags=re.S)
            for k, passage in enumerate(_split_body(body)):
                if len(passage) < 40:
                    continue
                chunks.append({
                    "id": f"doc:{project}#{anchor}" + (f"~{k}" if k else ""), "kind": "doc",
                    "title": f"{title} - {heading}", "text": passage, "project": project,
                    "url": f"{base_url}{project}/README.md" + (f"#{anchor}" if heading != "Overview" else ""),
                })
    return chunks


def build(base_url: str = "https://github.com/katherinekarow/portfolio/blob/main/professional-work/") -> dict:
    catalog, glossary, guard = load_yaml("catalog.yaml"), load_yaml("glossary.yaml"), load_yaml("guardrails.yaml")
    views = catalog["views"]
    g = validate_catalog(views)

    docs = []
    for v in views:
        cols = "; ".join(f"{c}: {d}" for c, d in v["columns"].items())
        docs.append({"id": f"view:{v['name']}", "kind": "view", "title": v["name"],
                     "text": f"{v['description']} Grain: {v['grain']}. Columns: {cols}. {v.get('sample_query', '')}"})
    for t in glossary["terms"]:
        docs.append({"id": f"term:{t['term']}", "kind": "term", "title": t["term"],
                     "text": " ".join([", ".join(t.get("aliases", [])), t["definition"], t.get("formula", ""), t.get("example", "")])})
    docs.extend(readme_chunks(ROOT.parent, base_url))

    synonyms = {}
    for t in glossary["terms"]:
        group = [t["term"].lower()] + [a.lower() for a in t.get("aliases", [])]
        for phrase in group:
            synonyms.setdefault(phrase, [])
            synonyms[phrase].extend(x for x in group if x != phrase and x not in synonyms[phrase])

    lineage = {n: {"upstream": sorted(nx.ancestors(g, n)), "downstream": sorted(nx.descendants(g, n)),
                   "direct_up": sorted(g.predecessors(n)), "direct_down": sorted(g.successors(n))} for n in g.nodes}
    for v in views:
        v.setdefault("sample_query", "")
    return {"views": views, "terms": glossary["terms"], "guardrails": guard, "docs": docs,
            "synonyms": synonyms, "lineage": lineage, "layers": LAYERS,
            "edges": sorted([list(e) for e in g.edges])}


def write(kb: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path
