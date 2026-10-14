"""Build ``manuscript/manuscript.md`` and ``manuscript/manuscript.docx`` from results files (rule 1).

The text lives in :mod:`sections` (one function per manuscript section). Every
number a section prints goes through :meth:`Ctx.num` (or a helper built on it),
which reads the value from a results/params/cohort file and records it in a
ledger; the ledger is written to ``manuscript/manuscript_numbers.csv`` (one row
per printed number: label, value, printed text, source file, key).

A lint pass then fails the build if any digit in the prose did not come from
the ledger. Allowed exceptions: citation numbers, Figure/Table/Section numbers,
"2D"/"3D", decision labels (D1-D15), lesion identifiers, model names (AT1,
P1/P2) and inline math between ``$...$``.

Usage: ``python manuscript/build_manuscript.py [--no-docx]``
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: E402,F401  (dtivl from this checkout)

OUT_MD = ROOT / "manuscript" / "manuscript.md"
OUT_DOCX = ROOT / "manuscript" / "manuscript.docx"
OUT_LEDGER = ROOT / "manuscript" / "manuscript_numbers.csv"
BIB = ROOT / "lit" / "references.bib"

# private-use markers around ledger numbers (stripped after the lint)
_M0, _M1, _M2 = "", "", ""


# ------------------------------------------------------------------ blocks
@dataclass
class Heading:
    text: str
    level: int = 1


@dataclass
class Para:
    text: str
    style: str | None = None        # None, "abstract", "keywords", "caption", "small"


@dataclass
class Figure:
    name: str                       # figures/<name>.png
    caption: str


@dataclass
class Table:
    name: str                       # results/tables/<name>.csv
    caption: str
    columns: list | None = None     # subset/order of csv columns
    rename: dict = field(default_factory=dict)
    formats: dict = field(default_factory=dict)   # column -> format spec
    note: str | None = None


class BuildError(RuntimeError):
    pass


# ------------------------------------------------------------------ context
class Ctx:
    """Data access, numbers ledger, citations and cross-references for the sections."""

    def __init__(self):
        self.ledger: list[dict] = []
        self._cache: dict = {}
        self.cite_order: list[str] = []
        self.bib = parse_bib(BIB)
        self.figures: dict[str, int] = {}
        self.tables: dict[str, int] = {}

    # ---- data
    def data(self, rel: str):
        """Parsed file (json -> dict/list, csv -> list of dict rows with numbers converted)."""
        if rel not in self._cache:
            path = ROOT / rel
            if not path.exists():
                raise FileNotFoundError(f"manuscript input missing: {rel}")
            if path.suffix == ".json":
                self._cache[rel] = json.loads(path.read_text())
            elif path.suffix == ".csv":
                import pandas as pd
                self._cache[rel] = pd.read_csv(path)
            else:
                raise BuildError(f"unsupported input type: {rel}")
        return self._cache[rel]

    def get(self, rel: str, key: str):
        """Value at a dotted key (``a.b.0.c``; integer segments index lists) of a json file."""
        x = self.data(rel)
        for seg in key.split(".") if key else []:
            if isinstance(x, list):
                x = x[int(seg)]
            elif isinstance(x, dict):
                if seg not in x:
                    raise KeyError(f"{rel}: no key {key!r} (missing {seg!r})")
                x = x[seg]
            else:
                raise KeyError(f"{rel}: cannot index {type(x).__name__} with {seg!r} in {key!r}")
        return x

    # ---- numbers
    def num(self, value, *, src: str, key: str, fmt: str = ".3g", label: str | None = None,
            scale: float = 1.0) -> str:
        """Format ``value * scale`` and record it; returns the marked text."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise BuildError(f"no value for {src}:{key} ({label or ''})")
        if isinstance(value, bool):
            raise BuildError(f"boolean printed as a number: {src}:{key}")
        v = value * scale if scale != 1.0 else value
        text = format(v, fmt) if fmt else str(v)
        text = text.replace("-", "−") if text.startswith("-") else text
        self.ledger.append({"id": len(self.ledger), "label": label or key, "value": repr(value),
                            "scale": scale, "printed": text, "source": src, "key": key})
        return f"{_M0}{len(self.ledger) - 1}{_M1}{text}{_M2}"

    def v(self, rel: str, key: str, fmt: str = ".3g", **kw) -> str:
        """Number at ``key`` of ``rel`` (json)."""
        return self.num(self.get(rel, key), src=rel, key=key, fmt=fmt, **kw)

    def count(self, value: int, *, src: str, key: str, label: str | None = None) -> str:
        return self.num(int(value), src=src, key=key, fmt="d", label=label)

    def pct(self, rel: str, key: str, nd: int = 1, scale: float = 1.0) -> str:
        return self.v(rel, key, fmt=f".{nd}f", scale=scale) + " %"

    def ci(self, lo_text: str, hi_text: str) -> str:
        return f"{lo_text}–{hi_text}"

    # ---- parameters and config (inputs, also recorded)
    def param(self, dotted: str, fmt: str = ".3g", scale: float = 1.0) -> str:
        from dtivl import params as P
        return self.num(P.get(dotted).value, src="params/study_config.yaml", key=dotted, fmt=fmt, scale=scale)

    def raw(self, dotted: str, fmt: str = ".3g") -> str:
        from dtivl import params as P
        return self.num(P.raw(dotted), src="params/study_config.yaml", key=dotted, fmt=fmt)

    def lit(self, row: str, fmt: str = ".3g", attr: str = "value") -> str:
        from dtivl import params as P
        q = P.lit(row)
        return self.num(getattr(q, attr), src="params/literature_parameters.csv", key=f"{row}.{attr}", fmt=fmt)

    # ---- citations and cross-references
    def cite(self, *keys: str) -> str:
        nums = []
        for k in keys:
            if k not in self.bib:
                raise BuildError(f"citation key not in references.bib: {k}")
            if k not in self.cite_order:
                self.cite_order.append(k)
            nums.append(self.cite_order.index(k) + 1)
        return "[" + _compress(sorted(set(nums))) + "]"

    def fig(self, name: str) -> str:
        self.figures.setdefault(name, len(self.figures) + 1)
        return f"Figure {self.figures[name]}"

    def tab(self, name: str) -> str:
        self.tables.setdefault(name, len(self.tables) + 1)
        return f"Table {self.tables[name]}"


def _compress(nums: list[int]) -> str:
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


# ------------------------------------------------------------------ bibliography
def parse_bib(path: Path) -> dict[str, dict]:
    """Minimal BibTeX reader for lit/references.bib (one field per line)."""
    out, cur = {}, None
    for line in path.read_text().splitlines():
        s = line.strip()
        m = re.match(r"@(\w+)\{([^,]+),", s)
        if m:
            cur = {"type": m.group(1).lower()}
            out[m.group(2).strip()] = cur
            continue
        m = re.match(r"(\w+)\s*=\s*\{(.*)\},?$", s)
        if cur is not None and m:
            cur[m.group(1).lower()] = m.group(2)
    return out


def _clean(s: str) -> str:
    s = re.sub(r"\{\\[\"'`^~](\w)\}", lambda m: m.group(1), s)
    s = s.replace("{", "").replace("}", "").replace("\\&", "&").replace("--", "–")
    return s


def format_reference(e: dict) -> str:
    """MDPI (ACS-based) style; authors 'Surname, I.; ...'; 'and others' -> 'et al.'."""
    authors = [a.strip() for a in _clean(e.get("author", "")).split(" and ")]
    etal = bool(authors) and authors[-1] == "others"
    authors = [a for a in authors if a and a != "others"]
    a_txt = "; ".join(authors) + ("; et al." if etal else "")
    parts = [a_txt + "." if a_txt and not a_txt.endswith(".") else a_txt, _clean(e.get("title", "")) + "."]
    venue = _clean(e.get("journal") or e.get("booktitle") or e.get("publisher") or e.get("howpublished") or "")
    year, vol, pages = e.get("year"), e.get("volume"), _clean(e.get("pages", ""))
    tail = " ".join(x for x in [f"*{venue}*" if venue else "", f"**{year}**," if year else "",
                                 f"*{vol}*," if vol else "", pages] if x).rstrip(",")
    if tail:
        parts.append(tail + ".")
    if e.get("doi"):
        parts.append(f"https://doi.org/{e['doi']}.")
    elif e.get("url"):
        parts.append(f"{e['url']}.")
    return " ".join(p for p in parts if p)


# ------------------------------------------------------------------ lint
_ALLOWED = [
    r"Figures? \d+(?:[a-z])?(?:\s*(?:and|,|–)\s*\d+[a-z]?)*",
    r"Tables? \d+(?:\s*(?:and|,|–)\s*\d+)*",
    r"Sections? \d+(?:\.\d+)*",
    r"\[\d+(?:[,–]\d+)*\]",
    r"\b[23]D\b", r"\bD\d{1,2}\b", r"\bAT1\b", r"\bP[12]\b", r"\bC\d\b",
    r"\b[CHPX](?:3D)?-A\d+(?:\.\d+)?-T\d+(?:\.\d+)?(?:-L\d+)?-[A-Z]+\b",
    r"\$[^$]*\$",
]
_ALLOWED_RE = re.compile("|".join(f"(?:{p})" for p in _ALLOWED))
_MARKED_RE = re.compile(f"{_M0}\\d+{_M1}.*?{_M2}", re.S)


def lint(text: str) -> list[str]:
    """Lines with digits that are neither ledger numbers nor allowed patterns."""
    bad = []
    for ln, line in enumerate(text.splitlines(), 1):
        if line.startswith("#") or line.startswith("!["):          # headings, image lines
            continue
        rest = _ALLOWED_RE.sub(" ", _MARKED_RE.sub(" ", line))
        if re.search(r"\d", rest):
            bad.append(f"{ln}: {line[:160]}")
    return bad


def strip_marks(text: str) -> str:
    return re.sub(f"{_M0}\\d+{_M1}(.*?){_M2}", r"\1", text, flags=re.S)


# ------------------------------------------------------------------ rendering
def _table_rows(ctx: Ctx, t: Table):
    df = ctx.data(f"results/tables/{t.name}.csv")
    cols = t.columns or list(df.columns)
    rows = []
    for _, r in df.iterrows():
        row = []
        for c in cols:
            x = r[c]
            if isinstance(x, float) and math.isnan(x):
                row.append("")
            elif c in t.formats and isinstance(x, (int, float)):
                row.append(format(x, t.formats[c]))
            else:
                row.append(str(x))
        rows.append(row)
    return [t.rename.get(c, c) for c in cols], rows


def render_md(ctx: Ctx, blocks: list) -> str:
    out = []
    for b in blocks:
        if isinstance(b, Heading):
            out += ["#" * (b.level + 1) + " " + b.text, ""]      # level 0 = title
        elif isinstance(b, Para):
            out += [b.text, ""]
        elif isinstance(b, Figure):
            n = ctx.figures.setdefault(b.name, len(ctx.figures) + 1)
            out += [f"![Figure {n}](../figures/{b.name}.png)", "", f"**Figure {n}.** {b.caption}", ""]
        elif isinstance(b, Table):
            n = ctx.tables.setdefault(b.name, len(ctx.tables) + 1)
            head, rows = _table_rows(ctx, b)
            out += [f"**Table {n}.** {b.caption}", "", "| " + " | ".join(head) + " |",
                    "|" + "---|" * len(head)]
            out += ["| " + " | ".join(r) + " |" for r in rows]
            out += [""] + ([b.note, ""] if b.note else [])
    return "\n".join(out)


def _add_runs(par, text: str):
    """Inline **bold**, *italic*, ^sup^ and ~sub~ for python-docx."""
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("*"):
            par.add_run(tok[1:-1]).italic = True
        elif tok.startswith("^"):
            par.add_run(tok[1:-1]).font.superscript = True
        elif tok.startswith("~"):
            par.add_run(tok[1:-1]).font.subscript = True
        else:
            par.add_run(tok)


def render_docx(ctx: Ctx, blocks: list, path: Path) -> None:
    from docx import Document
    from docx.shared import Inches, Pt

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Palatino Linotype", Pt(10)
    for b in blocks:
        if isinstance(b, Heading):
            doc.add_heading(strip_marks(b.text), level=min(b.level, 4) if b.level > 0 else 0)
        elif isinstance(b, Para):
            _add_runs(doc.add_paragraph(), strip_marks(b.text).replace("$", ""))
        elif isinstance(b, Figure):
            n = ctx.figures[b.name]
            img = ROOT / "figures" / f"{b.name}.png"
            if img.exists():
                doc.add_picture(str(img), width=Inches(6.3))
            p = doc.add_paragraph()
            p.add_run(f"Figure {n}. ").bold = True
            _add_runs(p, strip_marks(b.caption).replace("$", ""))
        elif isinstance(b, Table):
            n = ctx.tables[b.name]
            p = doc.add_paragraph()
            p.add_run(f"Table {n}. ").bold = True
            _add_runs(p, strip_marks(b.caption).replace("$", ""))
            head, rows = _table_rows(ctx, b)
            tb = doc.add_table(rows=1, cols=len(head))
            tb.style = "Table Grid"
            for c, h in zip(tb.rows[0].cells, head):
                c.text = h
            for r in rows:
                for c, x in zip(tb.add_row().cells, r):
                    c.text = x
            if b.note:
                doc.add_paragraph(strip_marks(b.note).replace("$", ""))
    doc.save(path)


def references_blocks(ctx: Ctx) -> list:
    blocks = [Heading("References", 1)]
    for i, k in enumerate(ctx.cite_order, 1):
        blocks.append(Para(f"{i}. {format_reference(ctx.bib[k])}", style="small"))
    return blocks


# ------------------------------------------------------------------ build
def build(docx: bool = True) -> dict:
    import sections

    ctx = Ctx()
    blocks = sections.document(ctx)
    body = render_md(ctx, blocks)
    bad = lint(body)                       # the reference list is outside the lint (years, volumes, pages)
    if bad:
        raise BuildError("numbers not taken from results files (rule 1):\n" + "\n".join(bad))
    blocks += references_blocks(ctx)
    md = render_md(ctx, blocks)
    for name in ctx.figures:
        if not (ROOT / "figures" / f"{name}.png").exists():
            raise BuildError(f"figure file missing: figures/{name}.png")
    banned = re.search(r"\bvalidat\w*", strip_marks(md), re.I)
    if banned:
        raise BuildError(f"rule 3: the word {banned.group(0)!r} appears in the manuscript")
    OUT_MD.write_text(strip_marks(md))
    with open(OUT_LEDGER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "value", "scale", "printed", "source", "key"])
        w.writeheader()
        w.writerows(ctx.ledger)
    if docx:
        render_docx(ctx, blocks, OUT_DOCX)
    return {"md": str(OUT_MD), "docx": str(OUT_DOCX) if docx else None, "n_numbers": len(ctx.ledger),
            "n_references": len(ctx.cite_order), "figures": ctx.figures, "tables": ctx.tables}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-docx", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(build(docx=not a.no_docx), indent=1))
    return 0


if __name__ == "__main__":
    # run through the importable module so that sections.py and this script share one set of block classes
    import build_manuscript
    raise SystemExit(build_manuscript.main())
