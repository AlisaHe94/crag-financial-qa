# Prompt Summary — Data Pipeline (`data_fetcher.py`, `document_processor.py`)

## Goal

A reproducible pipeline that downloads SEC EDGAR 10-K and 10-Q filings
for AAPL, MSFT, GOOGL, AMZN, META, parses the SEC `full-submission.txt`
archive, and splits each filing into modality-tagged chunks (text vs
real data tables).

## AI assistance

The team designed the pipeline architecture, selected the heuristics
(numeric-density rule, semantic chunking parameters, table representation
format), tested the output, and validated chunk quality on real
filings. AI was used as a coding assistant for implementation —
proposing function bodies, suggesting edge-case handling for SEC table
parsing, and iterating when our test cases surfaced bugs. The team
reviewed and modified all generated code before integration.

## High-level prompts used

These summarize what the team asked AI to draft; specific phrasing
during sessions varied.

- *Implement a data fetcher that downloads SEC EDGAR 10-K and 10-Q
  filings for AAPL, MSFT, GOOGL, AMZN, META using
  `sec_edgar_downloader`. Use the `SEC_EDGAR_USER_AGENT` env var.*
- *The downloaded files are SEC `full-submission.txt` archives.
  Extract the primary 10-K or 10-Q HTML document from the archive.*
- *Implement a `DocumentProcessor` that returns separate `text_chunks`
  and `table_chunks`. Use BeautifulSoup with lxml. Strip XBRL noise.
  Distinguish real data tables from layout tables using a
  numeric-density heuristic (≥3 cols, ≥20% numeric).*
- *For text, use sentence-embedding distance breakpoints (95th
  percentile) and greedy-merge for cosine > 0.85 to produce semantic
  chunks of 400–700 tokens.*
- *Convert each detected table into key-value sentence chunks. Use the
  year-row as canonical column keys for multi-row headers.
  Forward-fill blank header cells.*
- *For complex tables, try `pandas.read_html()` first; if the headers
  are garbage (Unnamed / numeric), fall back to manual key-value
  parsing.*
- *Merge adjacent symbol-only cells (`$`, `%`) into the next numeric
  cell so `[$, 416,161]` becomes `[$416,161]`.*
