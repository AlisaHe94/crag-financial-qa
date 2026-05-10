"""
SEC EDGAR data fetcher.
Downloads 10-K / 10-Q filings for specified tickers and fiscal years.
"""

from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

import requests
from sec_edgar_downloader import Downloader
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data/sec_filings"))

# Default tickers for the project demo
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_FILING_TYPES = ["10-K", "10-Q"]

# SEC requires a real contact email in the User-Agent string. Set
# SEC_EDGAR_USER_AGENT in .env to your email (or "Project Name <email>")
# rather than committing personal info to source.
SEC_USER_AGENT_EMAIL = os.getenv("SEC_EDGAR_USER_AGENT", "")


def download_filings(
    tickers: list[str] = DEFAULT_TICKERS,
    filing_types: list[str] = DEFAULT_FILING_TYPES,
    num_filings: int = 3,
    output_dir: Path = DATA_DIR,
) -> dict[str, list[Path]]:
    """Download SEC filings and return mapping of ticker -> list of file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not SEC_USER_AGENT_EMAIL:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT is not set. SEC requires a real email in the "
            "User-Agent header. Add it to .env, e.g.\n"
            "    SEC_EDGAR_USER_AGENT=you@example.com"
        )
    dl = Downloader("StatProject", SEC_USER_AGENT_EMAIL, str(output_dir))
    downloaded: dict[str, list[Path]] = {}

    for ticker in tickers:
        downloaded[ticker] = []
        for filing_type in filing_types:
            try:
                dl.get(filing_type, ticker, limit=num_filings)
                filing_dir = output_dir / "sec-edgar-filings" / ticker / filing_type
                if filing_dir.exists():
                    paths = list(filing_dir.rglob("*.htm")) + list(filing_dir.rglob("*.txt"))
                    downloaded[ticker].extend(paths)
                    logger.info(f"Downloaded {len(paths)} {filing_type} files for {ticker}")
                time.sleep(0.5)  # respect SEC rate limit (10 req/s)
            except Exception as e:
                logger.warning(f"Failed to download {filing_type} for {ticker}: {e}")

    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps({k: [str(p) for p in v] for k, v in downloaded.items()}, indent=2)
    )
    return downloaded


def _sec_headers() -> dict[str, str]:
    if not SEC_USER_AGENT_EMAIL:
        raise RuntimeError("SEC_EDGAR_USER_AGENT not set in .env")
    return {"User-Agent": f"StatProject {SEC_USER_AGENT_EMAIL}"}


def fetch_company_facts(ticker: str) -> Optional[dict]:
    """Fetch structured company facts from SEC EDGAR API (no key required)."""
    cik = _ticker_to_cik(ticker)
    if cik is None:
        return None
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    resp = requests.get(url, headers=_sec_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _ticker_to_cik(ticker: str) -> Optional[int]:
    """Resolve ticker to CIK using SEC EDGAR company tickers endpoint."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=_sec_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return int(entry["cik_str"])
    return None


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Download SEC EDGAR 10-K / 10-Q filings.")
    p.add_argument("-n", "--num-filings", type=int, default=1,
                   help="Most recent filings of each type per ticker (default 1; bump to 2-3 once pipeline is verified).")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                   help=f"Tickers to fetch (default: {' '.join(DEFAULT_TICKERS)}).")
    args = p.parse_args()

    results = download_filings(tickers=args.tickers, num_filings=args.num_filings)
    print()
    for ticker, paths in results.items():
        print(f"  {ticker}: {len(paths)} files")
