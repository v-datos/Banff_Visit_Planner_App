"""Hardened Tableau Public extractor entrypoint delegating to banff_planner.exogenous."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from banff_planner.config import BANFF_TABLEAU_WORKBOOK_URL
from banff_planner.exogenous import _fetch_tableau_workbook, fetch_banff_traffic_tableau

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def fetch_workbook(url: str = BANFF_TABLEAU_WORKBOOK_URL, attempts: int = 5):
    """Fetch Tableau Public workbook retrying rate limits and transient server errors."""
    return _fetch_tableau_workbook(url=url, attempts=attempts)


def main() -> None:
    path = fetch_banff_traffic_tableau(refresh=True)
    print(f"Data successfully extracted, validated (>=23h/day & 4x guard), and saved via {path}")


if __name__ == "__main__":
    main()
