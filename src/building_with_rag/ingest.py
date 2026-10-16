"""Story 2.1: turn the supplied BNS and IPC PDFs into section-level JSONL.

Run from the repository root:

    uv run python -m building_with_rag.ingest            # both acts
    uv run python -m building_with_rag.ingest --act bns  # one act
    uv run python -m building_with_rag.ingest --force    # rebuild even if up to date

Inputs (never modified): data/raw/*.pdf and data/raw/PROVENANCE.md.
Outputs: data/processed/bns_sections.jsonl and data/processed/ipc_sections.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
PROVENANCE = RAW_DIR / "PROVENANCE.md"

PARSER = "pypdf"

ACTS = {
    "bns": {
        "act": "BNS_2023",
        "act_label": "Bharatiya Nyaya Sanhita, 2023",
        "status": "in_force",
        "pdf": "BNS_2023_bare_act.pdf",
        "out": "bns_sections.jsonl",
        "prefix": "bns",
    },
    "ipc": {
        "act": "IPC_1860",
        "act_label": "The Indian Penal Code, 1860",
        "status": "repealed",
        "pdf": "IPC_1860_bare_act.pdf",
        "out": "ipc_sections.jsonl",
        "prefix": "ipc",
    },
}

FIELDS = [
    "section_id", "act", "act_label", "status", "chapter", "chapter_title", "section_number",
    "heading", "text", "source_pdf", "source_sha256", "parser", "parser_version",
    "source_status_version", "needs_review",
]

CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVXLC]+(?:\s?[A-Z])?)\s*$")
TERMINAL = (".", ";", ":", "—", "–", ",", "-")
MAX_HEADING_LINES = 3
# BNS: sub-chapter labels ("Of theft") sit between sections and are not part of any section.
SUBHEADING_RE = re.compile(r"^Of\s+\S")
# BNS: trailing page chrome after the last section.
BNS_NOISE_RE = re.compile(r"^(Note: For Reference only\.?|HomePage|[—–-]{3,})$")
# IPC: amendment footnote text that leaked into a section is flagged, never repaired.
FOOTNOTE_LEAK_RE = re.compile(
    r"(?:^|\s)\d{1,3}\.\s+(?:Subs\.|Ins\.|Omitted|Rep\.|Added|The words|The figures)"
)


class IngestError(Exception):
    pass


# --- inputs ---------------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_provenance(pdf_name: str) -> tuple[str, str | None, str | None]:
    """Return (expected sha256, retrieved date, status phrase) for a PDF from PROVENANCE.md."""
    if not PROVENANCE.exists():
        raise IngestError(f"missing input: {PROVENANCE}")
    text = PROVENANCE.read_text(encoding="utf-8")
    note = re.search(rf"`{re.escape(pdf_name)}` is.*?\*\*(in force|repealed)\*\*", text, re.S)
    status_phrase = note.group(1) if note else None
    for row in text.splitlines():
        if pdf_name in row and row.lstrip().startswith("|"):
            sha = re.search(r"`([0-9a-f]{64})`", row)
            date = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", row)
            if not sha:
                raise IngestError(f"no SHA-256 for {pdf_name} in PROVENANCE.md")
            return sha.group(1), date.group(1) if date else None, status_phrase
    raise IngestError(f"{pdf_name} is not listed in PROVENANCE.md")


# --- text cleaning --------------------------------------------------------------------------


def clean_pages(pages: list[str]) -> list[list[str]]:
    """Split pages into non-empty stripped lines and drop repeated headers/footers."""
    split = [[ln.strip() for ln in p.splitlines() if ln.strip()] for p in pages]

    # A line is a repeated header/footer if it sits in the first or last 3 lines of many pages.
    edge = Counter()
    for lines in split:
        edge.update({ln for ln in lines[:3] + lines[-3:] if len(ln) <= 80})
    threshold = max(5, int(len(split) * 0.3))
    repeated = {ln for ln, n in edge.items() if n >= threshold}

    cleaned: list[list[str]] = []
    pending_fn: int | None = None  # last footnote number on the previous page's footnote block
    for lines in split:
        lines = [ln for ln in lines if not re.fullmatch(r"\d{1,3}", ln)]  # bare page numbers
        if pending_fn is not None:
            lines = skip_footnote_continuation(lines, pending_fn + 1)
        pending_fn = None
        out: list[str] = []
        for i, ln in enumerate(lines):
            if re.fullmatch(r"-{10,}", ln):
                # The PDF can place section text after the separator, so keep non-footnote lines.
                body, pending_fn = split_footnote_block(lines[i + 1:])
                out.extend(body)
                break
            if ln in repeated and (i < 3 or i >= len(lines) - 3):
                continue
            out.append(ln)
        cleaned.append(out)
    return cleaned


def split_footnote_block(block: list[str]) -> tuple[list[str], int | None]:
    """Split lines after the footnote separator into (kept body lines, last footnote number).

    Footnotes are numbered 1, 2, 3, ... in order; a footnote runs until a line ends with a full
    stop or the next numbered footnote starts. Everything else is section text and is kept.
    """
    expected = 1
    body: list[str] = []
    in_footnote = False
    for ln in block:
        if ln.startswith(f"{expected}. "):
            expected += 1
            in_footnote = not ln.endswith(".")
        elif in_footnote:
            in_footnote = not ln.endswith(".")
        else:
            body.append(ln)
    return body, expected - 1 or None


def skip_footnote_continuation(lines: list[str], number: int) -> list[str]:
    """Drop footnotes that overflowed from the previous page onto the top of this one."""
    i = 0
    while i < len(lines) and lines[i].startswith(f"{number}. "):
        i += 1
        while i < len(lines) and not lines[i - 1].endswith(".") \
                and not lines[i].startswith(f"{number + 1}. "):
            i += 1  # wrapped footnote line
        number += 1
    return lines[i:]


def flatten(cleaned: list[list[str]]) -> list[tuple[str, int]]:
    return [(ln, pno) for pno, lines in enumerate(cleaned, start=1) for ln in lines]


def tag_chapters(lines: list[tuple[str, int]]) -> list[dict]:
    """Remove CHAPTER header lines and attach the current chapter to every other line."""
    out: list[dict] = []
    chapter, title = "", ""
    i = 0
    while i < len(lines):
        text, page = lines[i]
        m = CHAPTER_RE.match(text)
        if m:
            number = m.group(1).replace(" ", "")
            j = i + 1
            parts: list[str] = []
            while (
                j < len(lines)
                and len(parts) < 3
                and lines[j][0].isupper()  # whole line upper case: titles only, not body text
                and not CHAPTER_RE.match(lines[j][0])
            ):
                parts.append(lines[j][0])
                j += 1
            new_title = " ".join(parts)
            if number != chapter or (new_title and new_title != title):
                chapter, title = number, new_title or title
            i = j
            continue
        out.append({"text": text, "page": page, "chapter": chapter, "title": title})
        i += 1
    return out


# --- section splitting ----------------------------------------------------------------------


def find_heading_start(lines: list[dict], start: int, floor: int) -> tuple[int, bool]:
    """Walk back from a section line to find where its heading begins (BNS layout).

    A wrapped heading continues on lines that start in lower case, so the heading begins at the
    nearest line above that starts with a capital. Anything uncertain is flagged.
    """
    k = start - 1
    if k <= floor:
        return start, True
    first = k
    while (
        first - 1 > floor
        and start - first < MAX_HEADING_LINES
        and not lines[first]["text"][0].isupper()
    ):
        first -= 1
    before = lines[first - 1]["text"] if first - 1 > floor else ""
    flagged = (
        not lines[first]["text"][0].isupper()
        or start - first > 2
        or sum(len(lines[x]["text"]) for x in range(first, start)) > 220
        # the line above should end the previous section or be a sub-heading
        or bool(before and not before.endswith(TERMINAL) and not SUBHEADING_RE.match(before))
    )
    return first, flagged


def parse_bns(lines: list[dict]) -> list[dict]:
    """BNS layout: heading line(s), then `N. text` where N is the next section number."""
    body_start = next(
        (
            i
            for i, ln in enumerate(lines)
            if re.match(r"^1\.\s*\(1\)\s+This Act may be called", ln["text"])
        ),
        None,
    )
    if body_start is None:
        raise IngestError("BNS: could not find the start of section 1")
    lines = [ln for ln in lines[max(0, body_start - 4):] if not BNS_NOISE_RE.match(ln["text"])]

    starts: list[tuple[int, int, str]] = []  # (line index, number, text after "N. ")
    last = 0
    for i, ln in enumerate(lines):
        m = re.match(r"^(\d{1,3})\.(?!\d)\s*(\S.*)$", ln["text"])  # `115.(1)` has no space
        if m and last < int(m.group(1)) <= last + 3:
            starts.append((i, int(m.group(1)), m.group(2)))
            last = int(m.group(1))

    heading_starts: list[tuple[int, bool]] = []
    prev_idx = -1
    for idx, _, _ in starts:
        heading_starts.append(find_heading_start(lines, idx, prev_idx))
        prev_idx = idx

    def block_end(n: int) -> int:
        """End of section n's text: just before the next heading and any "Of ..." sub-heading."""
        if n + 1 >= len(starts):
            return len(lines)
        end = heading_starts[n + 1][0]
        while end - 1 > starts[n][0] and SUBHEADING_RE.match(lines[end - 1]["text"]) \
                and not lines[end - 1]["text"].endswith(TERMINAL):
            end -= 1
        return end

    sections: list[dict] = []
    prev_number = 0
    for n, (idx, number, first_text) in enumerate(starts):
        end = block_end(n)
        h_start, h_flag = heading_starts[n]
        body = [first_text] + [lines[x]["text"] for x in range(idx + 1, end)]
        heading = " ".join(lines[x]["text"] for x in range(h_start, idx))
        sections.append({
            "label": str(number),
            "number": number,
            "chapter": lines[idx]["chapter"],
            "chapter_title": lines[idx]["title"],
            "heading": heading,
            "text": " ".join(body),
            "page_start": lines[idx]["page"],
            "page_end": lines[end - 1]["page"],
            "review": h_flag or number != prev_number + 1 or not heading,
        })
        prev_number = number
    return sections


# Amended sections are printed inside a footnote bracket, e.g. `8*[4. Heading.--text`.
# Repealed sections have no body, so a few numbers in a row can be absent.
IPC_MAX_GAP = 8
IPC_START_RE = re.compile(r"^(?:\d{0,3}\*+\[\s*)*(\d{1,3}[A-Z]{0,2})\.\s+(\S.*)$")


def starts_section(text: str, label: str) -> bool:
    m = IPC_START_RE.match(text)
    return bool(m) and m.group(1) == label


def parse_ipc(lines: list[dict]) -> list[dict]:
    """IPC layout: `N. Heading.--text`, preceded by a duplicate `N.` + heading index entry."""
    # Drop the duplicated index entry: a bare `N.` line plus heading lines before `N. Heading.--`.
    kept: list[dict] = []
    i = 0
    while i < len(lines):
        m = re.fullmatch(r"(\d{1,3}[A-Z]{0,2})\.", lines[i]["text"])
        if m:
            j = i + 1
            label = m.group(1)
            while j < len(lines) and j - i <= 6 and not starts_section(lines[j]["text"], label):
                j += 1
            if j < len(lines) and starts_section(lines[j]["text"], label):
                i = j
                continue
        kept.append(lines[i])
        i += 1
    lines = kept

    def split_start(idx: int) -> tuple[str, str, str, int] | None:
        """Return (label, heading, remaining text, lines consumed) if idx starts a section."""
        m = IPC_START_RE.match(lines[idx]["text"])
        if not m:
            return None
        joined = m.group(2)
        used = 1
        while True:
            hm = re.match(r"^(.+?)\.?\s*--\s*(.*)$", joined)
            if hm:
                return m.group(1), hm.group(1).strip(), hm.group(2), used
            nxt = idx + used
            if used > 5 or nxt >= len(lines) or IPC_START_RE.match(lines[nxt]["text"]):
                return None
            joined += " " + lines[nxt]["text"]
            used += 1

    starts: list[tuple[int, str, str, str, int]] = []
    seen: set[str] = set()
    last_int = 0
    for i in range(len(lines)):
        s = split_start(i)
        if not s:
            continue
        label, heading, rest, used = s
        number = int(re.match(r"\d+", label).group())
        if label in seen or not (last_int <= number <= last_int + IPC_MAX_GAP):
            continue
        seen.add(label)
        last_int = number
        starts.append((i, label, heading, rest, used))

    sections: list[dict] = []
    prev_number = 0
    for n, (idx, label, heading, rest, used) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        body = [rest] + [lines[x]["text"] for x in range(idx + used, end)]
        number = int(re.match(r"\d+", label).group())
        sections.append({
            "label": label,
            "number": number,
            "chapter": lines[idx]["chapter"],
            "chapter_title": lines[idx]["title"],
            "heading": heading,
            "text": " ".join(body),
            "page_start": lines[idx]["page"],
            "page_end": lines[end - 1]["page"],
            "review": label != str(number) or number not in (prev_number, prev_number + 1),
        })
        prev_number = number
    return sections


# --- records and output ---------------------------------------------------------------------


def build_records(key: str, sections: list[dict], sha: str, status_version: str | None,
                  blank_pages: set[int]) -> list[dict]:
    cfg = ACTS[key]
    records = []
    for s in sections:
        text = re.sub(r"\s+", " ", s["text"]).strip()
        review = (
            s["review"]
            or status_version is None
            or not s["chapter"]
            or not s["heading"]
            or bool(FOOTNOTE_LEAK_RE.search(text))
            or any(p in blank_pages for p in range(s["page_start"], s["page_end"] + 1))
        )
        rec = {
            "section_id": f"{cfg['prefix']}:{s['label']}",
            "act": cfg["act"],
            "act_label": cfg["act_label"],
            "status": cfg["status"],
            "chapter": s["chapter"],
            "chapter_title": s["chapter_title"],
            "section_number": s["number"],
            "heading": re.sub(r"\s+", " ", s["heading"]).strip(),
            "text": text,
            "source_pdf": cfg["pdf"],
            "source_sha256": sha,
            "parser": PARSER,
            "parser_version": version(PARSER),
            "source_status_version": status_version,
            "needs_review": bool(review),
            # extra extraction data (not a substitute for the fields above)
            "section_label": s["label"],
            "page_start": s["page_start"],
            "page_end": s["page_end"],
        }
        records.append(rec)
    return records


def check_records(key: str, records: list[dict]) -> None:
    if not records:
        raise IngestError(f"{key}: no sections extracted")
    ids = [r["section_id"] for r in records]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise IngestError(f"{key}: duplicate section_id values: {dupes[:5]}")
    empty = [r["section_id"] for r in records if not r["text"]]
    if empty:
        raise IngestError(f"{key}: empty section text for {empty[:5]}")


def already_current(path: Path, sha: str) -> bool:
    if not path.exists():
        return False
    try:
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln]
    except json.JSONDecodeError:
        return False
    return bool(rows) and all(
        r.get("source_sha256") == sha and r.get("parser_version") == version(PARSER) for r in rows
    )


def write_replace(path: Path, records: list[dict]) -> None:
    """Write the whole file to a temp path, then replace the old one in a single step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def ingest(key: str, force: bool) -> None:
    cfg = ACTS[key]
    pdf = RAW_DIR / cfg["pdf"]
    if not pdf.exists():
        raise IngestError(f"missing input: {pdf}")
    expected_sha, retrieved, status_phrase = read_provenance(cfg["pdf"])
    sha = sha256_of(pdf)
    if sha != expected_sha:
        raise IngestError(
            f"{cfg['pdf']}: SHA-256 {sha} does not match PROVENANCE.md {expected_sha}"
        )

    out = OUT_DIR / cfg["out"]
    if not force and already_current(out, sha):
        print(f"{key}: {out.relative_to(ROOT)} already current for this source; left unchanged")
        return

    pages = [(p.extract_text() or "") for p in PdfReader(str(pdf)).pages]
    blank_pages = {i for i, t in enumerate(pages, start=1) if len(t.strip()) < 20}
    if blank_pages:
        print(f"{key}: WARNING pages with no extractable text (scanned?): {sorted(blank_pages)}")

    lines = tag_chapters(flatten(clean_pages(pages)))
    sections = parse_bns(lines) if key == "bns" else parse_ipc(lines)

    # Use the status and retrieval date as supplied; if they are missing or disagree, do not guess.
    status_ok = status_phrase == cfg["status"].replace("_", " ")
    status_version = f"{status_phrase}; retrieved {retrieved}" if status_ok and retrieved else None
    records = build_records(key, sections, sha, status_version, blank_pages)
    check_records(key, records)
    write_replace(out, records)

    flagged = [r["section_id"] for r in records if r["needs_review"]]
    numbers = {r["section_number"] for r in records}
    missing = [n for n in range(1, max(numbers) + 1) if n not in numbers]
    print(f"{key}: wrote {len(records)} records to {out.relative_to(ROOT)}")
    print(f"{key}: section numbers with no record in 1..{max(numbers)}: {missing}")
    print(f"{key}: {len(flagged)} flagged needs_review: {flagged}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--act", choices=["bns", "ipc", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="rebuild even if output is current")
    args = ap.parse_args()
    keys = list(ACTS) if args.act == "all" else [args.act]
    status = 0
    for key in keys:
        try:
            ingest(key, args.force)
        except IngestError as e:
            print(f"{key}: ERROR {e}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
