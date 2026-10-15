"""Small helpers for working on SQL text without a full parser.

`mask` hides string literals and comments (same length, newlines kept) so searches for
keywords, commas and parentheses only ever see code. All positions found on the masked
text are valid in the original text.
"""
from __future__ import annotations

import re

_LITERAL_OR_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/|'(?:''|[^'])*'", re.S)


def mask(sql: str) -> str:
    out = list(sql)
    for m in _LITERAL_OR_COMMENT.finditer(sql):
        is_string = m.group(0).startswith("'")
        for i in range(m.start(), m.end()):
            if out[i] == "\n":
                continue
            out[i] = ("'" if i in (m.start(), m.end() - 1) else "x") if is_string else " "
    return "".join(out)


def matching_paren(masked: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(masked)):
        if masked[i] == "(":
            depth += 1
        elif masked[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced parenthesis at {open_idx}")


def split_top(text: str, sep: str = ",") -> list[str]:
    """Split on `sep` at parenthesis depth 0 (ignoring separators inside strings/comments)."""
    m = mask(text)
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(m):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def split_and(text: str) -> list[str]:
    """Split a boolean expression on top-level AND (not inside parentheses or BETWEEN x AND y)."""
    m = mask(text)
    depth, start, parts, i = 0, 0, [], 0
    between_pending = False
    while i < len(m):
        ch = m[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and re.match(r"\bbetween\b", m[i:i + 8], re.I) and (i == 0 or not m[i - 1].isalnum()):
            between_pending = True
        elif depth == 0 and re.match(r"and\b", m[i:i + 4], re.I) and (i == 0 or not (m[i - 1].isalnum() or m[i - 1] == "_")):
            if between_pending:
                between_pending = False
            else:
                parts.append(text[start:i])
                start = i + 3
        i += 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def find_top_keyword(masked: str, keyword: str, start: int = 0) -> int:
    """Position of a keyword at depth 0 (relative to `start`), or -1."""
    depth = 0
    kw = re.compile(rf"\b{keyword}\b", re.I)
    for i in range(start, len(masked)):
        ch = masked[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and kw.match(masked, i) and (i == 0 or not (masked[i - 1].isalnum() or masked[i - 1] == "_")):
            return i
    return -1


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1
