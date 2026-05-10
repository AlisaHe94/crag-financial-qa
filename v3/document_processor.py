"""
Table-aware document processor with MultiFinRAG-style semantic chunking.

Semantic chunking algorithm (Gondhalekar et al., 2025):
  1. Sentence segmentation
  2. Embed each sentence with BAAI/bge-base-en-v1.5
  3. Compute cosine-distance between adjacent sentences: d_j = 1 - cos(e_j, e_{j+1})
  4. Mark breakpoints where d_j > 95th percentile of {d_j}
  5. Split at breakpoints; greedily merge chunks with cosine similarity > 0.85
  6. Separate FAISS indexes for text / table / image chunks (modality-aware)

Two modes for ablation:
  table_aware=False  — naive fixed-size chunking, no table parsing (baseline)
  table_aware=True   — full semantic + table extraction (proposed system)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
Modality = Literal["text", "table", "image"]


@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_id: int
    modality: Modality = "text"
    is_table: bool = False          # kept for backward compat
    page_num: Optional[int] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(
    file_path: str | Path,
    table_aware: bool = True,
    chunk_size: int = 512,          # used only in naive mode
    chunk_overlap: int = 64,        # used only in naive mode
    _embedder: Optional[SentenceTransformer] = None,
) -> list[DocumentChunk]:
    """Parse a document and return a list of chunks (text + table modalities)."""
    path = Path(file_path)
    embedder = _embedder  # caller may share one instance for efficiency

    if path.suffix.lower() == ".pdf":
        return _parse_pdf(path, table_aware, chunk_size, chunk_overlap, embedder)
    return _parse_text_file(path, table_aware, chunk_size, chunk_overlap, embedder)


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def _parse_pdf(
    path: Path,
    table_aware: bool,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Optional[SentenceTransformer],
) -> list[DocumentChunk]:
    all_text: list[str] = []
    table_chunks: list[DocumentChunk] = []
    chunk_id = 0

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if table_aware:
                t_chunks, used_bboxes = _extract_tables(page, page_num, str(path), chunk_id)
                table_chunks.extend(t_chunks)
                chunk_id += len(t_chunks)
                text = _text_excluding_bboxes(page, used_bboxes)
            else:
                text = page.extract_text() or ""
            if text.strip():
                all_text.append(text)

    raw_text = " ".join(all_text)
    if table_aware and embedder is not None:
        text_chunks = _semantic_chunk(raw_text, str(path), chunk_id, embedder)
    else:
        text_chunks = _naive_chunk(raw_text, chunk_size, chunk_overlap, str(path), chunk_id)

    return table_chunks + text_chunks


# ---------------------------------------------------------------------------
# Table extraction → Markdown
# ---------------------------------------------------------------------------

def _extract_tables(
    page, page_num: int, source: str, start_id: int
) -> tuple[list[DocumentChunk], list[tuple]]:
    chunks: list[DocumentChunk] = []
    bboxes: list[tuple] = []

    for table_obj in page.find_tables():
        bbox = table_obj.bbox
        bboxes.append(bbox)
        rows = table_obj.extract()
        if not rows:
            continue
        md = _rows_to_kv_sentences(rows)  # was _rows_to_markdown — easier for 8B LLM
        chunks.append(
            DocumentChunk(
                text=md,
                source=source,
                chunk_id=start_id + len(chunks),
                modality="table",
                is_table=True,
                page_num=page_num,
                metadata={"bbox": bbox},
            )
        )
    return chunks, bboxes


def _rows_to_markdown(rows: list[list]) -> str:
    """Render a table as Markdown pipes (kept for backward compat; ablation
    showed Llama-8B struggles to extract values from this format)."""
    if not rows:
        return ""

    def cell(v) -> str:
        return str(v).replace("|", "\\|").strip() if v is not None else ""

    header = rows[0]
    lines = ["| " + " | ".join(cell(c) for c in header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


def _html_table_to_kv_via_pandas(table_el) -> str:
    """Use `pandas.read_html` to convert one HTML <table> to KV sentences.

    Why pandas over our manual parsing: SEC tables routinely use rowspan
    and colspan in their <thead>, which our naive `_html_table_to_rows`
    doesn't track. pandas.read_html handles them natively, producing a
    DataFrame whose columns may be a MultiIndex (super-header above sub-
    header) and whose data cells correctly inherit values from spanning
    parents. We then walk the DataFrame and emit one
    `{row label}, {col header}: {value}` line per cell.

    Returns "" on any pandas error so the caller can fall back to
    `_rows_to_kv_sentences` (the manual path).
    """
    try:
        import pandas as pd
        from io import StringIO
        # keep_default_na=False so empty cells become "" not NaN.
        dfs = pd.read_html(StringIO(str(table_el)), keep_default_na=False)
    except Exception as e:
        logger.debug(f"pandas.read_html failed on table: {e}")
        return ""

    if not dfs:
        return ""
    df = dfs[0]
    if df.empty or len(df.columns) < 2:
        return ""

    # Extract a flat header list. If columns is a MultiIndex (multi-row
    # header in the source HTML), pick the most specific (last) non-empty
    # level; otherwise use the column name directly. "Unnamed: N" is what
    # pandas inserts for header cells that were truly empty in the source —
    # treat those as empty.
    def _level_to_header(col) -> str:
        if isinstance(col, tuple):
            for level in reversed(col):
                level_str = str(level).strip()
                if level_str and not level_str.startswith("Unnamed"):
                    return level_str
            return ""
        col_str = str(col).strip()
        if col_str.startswith("Unnamed"):
            return ""
        return col_str

    raw_headers = [_level_to_header(c) for c in df.columns]

    # Sanity check: if pandas couldn't infer real headers (column names
    # are all digits or "Unnamed: N"), fall back to manual parsing.
    # Otherwise we end up with garbage like "Family of Apps, 4: 102469".
    real_headers = sum(
        1 for h in raw_headers
        if h and not h.isdigit() and not h.lower().startswith("unnamed")
    )
    if real_headers < 2:
        logger.debug(
            f"pandas.read_html produced only numeric/unnamed headers "
            f"({raw_headers[:5]}…); falling back to manual KV"
        )
        return ""

    # Forward-fill: empty header cell inherits the most recent non-empty
    # one to its left (handles a year column with adjacent currency-symbol
    # column whose header was blank).
    headers: list[str] = []
    last = ""
    for h in raw_headers:
        if h:
            last = h
        headers.append(last)

    sentences: list[str] = []
    for _, row in df.iterrows():
        # Row label = first cell of the row (assumed to be text)
        row_label = str(row.iloc[0]).strip()
        if not row_label or row_label.lower() == "nan":
            continue
        for col_idx in range(1, len(row)):
            value = str(row.iloc[col_idx]).strip()
            if not value or value.lower() == "nan":
                continue
            # Skip dashes / placeholder no-data markers.
            if value in ("—", "–", "-"):
                continue
            col_label = headers[col_idx] if col_idx < len(headers) else ""
            if col_label:
                sentences.append(f"{row_label}, {col_label}: {value}")
            else:
                sentences.append(f"{row_label}: {value}")

    return "\n".join(sentences)


def _rows_to_kv_sentences(rows: list[list]) -> str:
    """Convert a 2-D table to plain English key-value sentences.

    Strategy (key fix vs prior version):
      1. Find the header row by counting year-like substrings (regex catches
         "2025" inside "September 27, 2025" too, not only pure 4-digit cells).
      2. Take only the NON-EMPTY headers from that row → N labels.
      3. For each data row, MERGE symbol cells ($, (, %, ) — see
         _merge_symbol_cells) with their adjacent value cells. This collapses
         11-cell data rows like ["Family of Apps","$","102,469","$","87,109",
         "$","62,871","18","%","39","%"] down to 5 merged values:
         ["$102,469","$87,109","$62,871","18%","39%"].
      4. Map the i-th merged value to the i-th non-empty header label.
         This is what makes the previous failure mode (header has 6 cells,
         data has 11, indices misalign) go away entirely.

    Falls back to Markdown if the table doesn't fit the assumed layout.
    """
    if not rows or len(rows) < 2:
        return _rows_to_markdown(rows)

    def s(v) -> str:
        return str(v).strip() if v is not None else ""

    year_re = re.compile(r"\b(20\d{2}|19\d{2})\b")

    def _year_count(row) -> int:
        years = set()
        for c in row:
            for m in year_re.findall(s(c)):
                years.add(m)
        return len(years)

    # Pick the header row: scan the first 5 rows, take the one with the
    # most distinct year mentions. Ties broken by earlier index.
    header_row_idx = 0
    best_score = 0
    for i in range(min(5, len(rows))):
        score = _year_count(rows[i])
        if score > best_score:
            best_score = score
            header_row_idx = i

    # Non-empty header labels in left-to-right order.
    non_empty_headers = [s(c) for c in rows[header_row_idx] if s(c)]
    if not non_empty_headers:
        return _rows_to_markdown(rows)

    def _merge_symbol_cells(cells) -> list[str]:
        """Merge stray '$' / '(' (prefix) and '%' / ')' (suffix) standalone
        cells with their adjacent value cells, dropping empty cells along
        the way. Returns the cleaned list of value strings."""
        out: list[str] = []
        pending_prefix = ""
        for c in cells:
            c_str = s(c)
            if not c_str:
                continue
            if c_str in ("$", "("):
                pending_prefix += c_str
                continue
            if c_str in ("%", ")"):
                if out:
                    out[-1] = out[-1] + c_str
                continue
            if c_str in ("—", "–", "-", "n/a", "N/A"):
                continue
            full = pending_prefix + c_str
            pending_prefix = ""
            out.append(full)
        return out

    sentences: list[str] = []
    for row in rows[header_row_idx + 1:]:
        if not row:
            continue
        row_label = s(row[0])
        if not row_label:
            continue
        merged_values = _merge_symbol_cells(row[1:])
        for i, value in enumerate(merged_values):
            header = non_empty_headers[i] if i < len(non_empty_headers) else ""
            if header:
                sentences.append(f"{row_label}, {header}: {value}")
            else:
                sentences.append(f"{row_label}: {value}")

    if not sentences:
        return _rows_to_markdown(rows)
    return "\n".join(sentences)


def _text_excluding_bboxes(page, bboxes: list[tuple]) -> str:
    if not bboxes:
        return page.extract_text() or ""
    words = page.extract_words()
    kept = [
        w["text"] for w in words
        if not any(
            bx0 <= w["x0"] and w["top"] >= by0 and w["x1"] <= bx1 and w["bottom"] <= by1
            for bx0, by0, bx1, by1 in bboxes
        )
    ]
    return " ".join(kept)


# ---------------------------------------------------------------------------
# Semantic chunking (MultiFinRAG §3.3.2)
# ---------------------------------------------------------------------------

def _semantic_chunk(
    text: str,
    source: str,
    start_id: int,
    embedder: SentenceTransformer,
    merge_threshold: float = 0.85,
    breakpoint_percentile: int = 95,
) -> list[DocumentChunk]:
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return [DocumentChunk(text=text, source=source, chunk_id=start_id, modality="text")]

    # Embed all sentences
    embeddings = embedder.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

    # Cosine distance between adjacent sentences
    distances = np.array([
        1 - float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(sentences) - 1)
    ])

    threshold = np.percentile(distances, breakpoint_percentile)
    breakpoints = set(i + 1 for i, d in enumerate(distances) if d > threshold)

    # Split into initial chunks at breakpoints
    raw_chunks: list[str] = []
    current: list[str] = []
    for i, sent in enumerate(sentences):
        if i in breakpoints and current:
            raw_chunks.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
    if current:
        raw_chunks.append(" ".join(current))

    # Merge near-duplicate chunks (cosine similarity > merge_threshold)
    merged = _merge_similar_chunks(raw_chunks, embedder, merge_threshold)

    return [
        DocumentChunk(
            text=c,
            source=source,
            chunk_id=start_id + i,
            modality="text",
        )
        for i, c in enumerate(merged)
    ]


def _merge_similar_chunks(
    chunks: list[str], embedder: SentenceTransformer, threshold: float
) -> list[str]:
    if len(chunks) <= 1:
        return chunks
    embeddings = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    merged: list[str] = [chunks[0]]
    merged_embs = [embeddings[0]]
    for i in range(1, len(chunks)):
        # Compare against last merged chunk
        sim = float(np.dot(embeddings[i], merged_embs[-1]))
        if sim > threshold:
            merged[-1] = merged[-1] + " " + chunks[i]
            # Re-embed merged chunk
            merged_embs[-1] = embedder.encode([merged[-1]], normalize_embeddings=True)[0]
        else:
            merged.append(chunks[i])
            merged_embs.append(embeddings[i])
    return merged


# ---------------------------------------------------------------------------
# Naive fixed-size chunking (baseline / ablation)
# ---------------------------------------------------------------------------

def _naive_chunk(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    source: str,
    start_id: int,
) -> list[DocumentChunk]:
    words = text.split()
    chunks: list[DocumentChunk] = []
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i: i + chunk_size])
        if chunk_text.strip():
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source=source,
                    chunk_id=start_id + len(chunks),
                    modality="text",
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Plain-text / HTML file parsing
# ---------------------------------------------------------------------------

def _parse_text_file(
    path: Path,
    table_aware: bool,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Optional[SentenceTransformer],
) -> list[DocumentChunk]:
    """Parse .txt or .htm/.html files.

    For HTML (the SEC EDGAR case), we use BeautifulSoup to:
      * drop <script>/<style>/<head> noise (CSS, JS, XBRL meta)
      * extract <table> elements as separate `modality="table"` chunks
        (only when table_aware=True; baseline drops them)
      * chunk the remaining body text via the same semantic / naive paths
        used by the PDF code path, so HTM and PDF behave symmetrically.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")

    # ---- SEC submission archive (.txt) -------------------------------------
    # sec-edgar-downloader v5 saves filings as `full-submission.txt`, a
    # multi-part SEC archive that wraps the primary HTML inside <DOCUMENT>
    # blocks. Detect that envelope and extract the primary 10-K/10-Q HTML so
    # we can run our normal HTML pipeline on it.
    if path.suffix.lower() not in (".htm", ".html"):
        if "<SEC-DOCUMENT" in raw[:5000] or "<DOCUMENT>" in raw[:10000]:
            primary_html = _extract_primary_doc_from_sec_archive(raw)
            if primary_html is not None:
                logger.info(f"  {path.name}: extracted primary doc from SEC archive ({len(primary_html):,} chars)")
                raw = primary_html
                # Fall through to the HTML parser below.
            else:
                logger.warning(f"  {path.name}: SEC archive but no 10-K/10-Q DOCUMENT block found")
                return []
        else:
            # Genuine plain text — same behavior as before.
            cleaned = re.sub(r"\s+", " ", raw).strip()
            if table_aware and embedder is not None:
                return _semantic_chunk(cleaned, str(path), 0, embedder)
            return _naive_chunk(cleaned, chunk_size, chunk_overlap, str(path), 0)

    # ---- HTML (SEC EDGAR 10-K / 10-Q) --------------------------------------
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError(
            "beautifulsoup4 is required for parsing SEC .htm filings. "
            "Run:  pip install -r requirements.txt"
        ) from e

    soup = BeautifulSoup(raw, "lxml")

    # Drop noise wholesale: <style>, <script>, <head>/meta, hidden XBRL refs.
    for tag in soup.find_all(["script", "style", "noscript", "head", "meta", "link"]):
        tag.decompose()
    # SEC inline-XBRL <ix:hidden> blocks contain machine-readable duplicates
    # of values that already appear elsewhere; safe to drop.
    for tag in soup.find_all(lambda t: t.name and t.name.startswith("ix:hidden")):
        tag.decompose()

    table_chunks: list[DocumentChunk] = []
    chunk_id = 0

    if table_aware:
        # SEC HTML uses <table> for two very different purposes:
        #   (a) financial data tables (income statement, balance sheet, etc.)
        #   (b) page-layout containers wrapping entire narrative sections
        # We only want to extract (a) as structured chunks. (b) must stay in
        # the soup so its narrative text is captured in the body. We use a
        # numeric-density heuristic to tell them apart.
        for table_el in soup.find_all("table"):
            rows = _html_table_to_rows(table_el)
            if not _looks_like_data_table(rows):
                continue  # leave layout tables in place — get_text() will flatten them
            # Try pandas.read_html first (handles rowspan/colspan natively, so
            # SEC's multi-row "Year Ended | 2025 | 2024 | 2023" headers align
            # properly). Fall back to our manual KV converter if pandas can't
            # parse this particular table.
            md = _html_table_to_kv_via_pandas(table_el)
            if not md or len(md.strip()) < 50:
                md = _rows_to_kv_sentences(rows)
            if len(md.strip()) >= 50:
                table_chunks.append(
                    DocumentChunk(
                        text=md,
                        source=str(path),
                        chunk_id=chunk_id,
                        modality="table",
                        is_table=True,
                    )
                )
                chunk_id += 1
            # Remove the data table from soup so its raw cells don't also
            # pollute the body text (which would compete with the structured
            # chunk during retrieval — exactly what the modality-aware index
            # is designed to avoid).
            table_el.decompose()
    # Baseline (table_aware=False): do NOT decompose any tables. Naive RAG
    # treats tables as flat text — letting get_text() walk them gives us the
    # cell content with spaces between, which is the desired baseline behavior.

    body_text = soup.get_text(separator=" ")
    body_text = re.sub(r"\s+", " ", body_text).strip()

    if table_aware and embedder is not None:
        text_chunks = _semantic_chunk(body_text, str(path), chunk_id, embedder)
    else:
        text_chunks = _naive_chunk(body_text, chunk_size, chunk_overlap, str(path), chunk_id)

    return table_chunks + text_chunks


def _html_table_to_rows(table_el) -> list[list[str]]:
    """Convert a BeautifulSoup <table> element into rows-of-strings.

    Naive: ignores rowspan/colspan. Sufficient for most SEC financial tables,
    which are dense grids without complex cell merging. If we hit rowspan-heavy
    tables that produce garbage, swap this for pandas.read_html(str(table_el)).
    """
    rows: list[list[str]] = []
    for tr in table_el.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        rows.append([c.get_text(separator=" ", strip=True) for c in cells])
    return rows


def _extract_primary_doc_from_sec_archive(raw: str) -> Optional[str]:
    """Pull the primary 10-K/10-Q HTML out of a SEC `full-submission.txt`.

    Format:
        <SEC-DOCUMENT>...
        <SEC-HEADER>...</SEC-HEADER>
        <DOCUMENT>
        <TYPE>10-K
        <SEQUENCE>1
        <FILENAME>aapl-20230930.htm
        <DESCRIPTION>10-K
        <TEXT>
        <html>...inline-XBRL HTML body...</html>
        </TEXT>
        </DOCUMENT>
        <DOCUMENT>  ... exhibits ...  </DOCUMENT>

    Returns the HTML inside the first <DOCUMENT> block whose TYPE matches
    10-K, 10-Q, 10-K/A, or 10-Q/A. Exhibits (EX-21.1, EX-99.1, etc.) are
    skipped. Returns None if no matching document is found.
    """
    target_types = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
    doc_re = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE)
    type_re = re.compile(r"<TYPE>([^\s<]+)", re.IGNORECASE)
    text_re = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL | re.IGNORECASE)

    for m in doc_re.finditer(raw):
        block = m.group(1)
        type_match = type_re.search(block)
        if not type_match:
            continue
        if type_match.group(1).strip().upper() not in target_types:
            continue
        text_match = text_re.search(block)
        if text_match:
            return text_match.group(1)
    return None


_NUMERIC_CELL_RE = re.compile(r"^[-+(]?\$?\s*[\d,]+(?:\.\d+)?\s*[)%]?\s*$")


def _looks_like_data_table(rows: list[list[str]]) -> bool:
    """Heuristic to distinguish financial data tables from layout-only tables.

    SEC HTML uses <table> for both. A real data table is dense, has multiple
    columns, and many cells look like numbers (e.g. 1,234, $5.6, (78)%, 90.0).
    A layout table typically has 1-2 columns of long prose.

    We classify as a "data table" when:
      * at least 2 rows and 3 columns
      * at least 20% of non-empty cells match a numeric pattern
    """
    if len(rows) < 2:
        return False
    n_cols = max((len(r) for r in rows), default=0)
    if n_cols < 3:
        return False
    n_total = 0
    n_numeric = 0
    for row in rows:
        for cell in row:
            stripped = (cell or "").strip()
            if not stripped:
                continue
            n_total += 1
            if _NUMERIC_CELL_RE.match(stripped):
                n_numeric += 1
    if n_total == 0:
        return False
    return (n_numeric / n_total) >= 0.20


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]
