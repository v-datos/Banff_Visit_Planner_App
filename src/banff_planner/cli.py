"""Command-line interface for running the Banff Visit Planner forecasting & site-build pipeline."""

from __future__ import annotations

import argparse
import logging

from banff_planner.config import DEFAULT_HORIZON_DAYS
from banff_planner.pipeline import run_full_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="banff-planner",
        description="Banff Visit Planner: daily exogenous forecasting, conformal intervals, and static web builder.",
    )
    parser.add_argument(
        "--refresh-web",
        action="store_true",
        help="Fetch fresh Tableau traffic + live ECCC/Open-Meteo/BoC/StatCan/Alberta feeds before modeling.",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Reuse cached expanding-window CV metrics/OOF residuals if present (faster local iteration).",
    )
    parser.add_argument(
        "--no-site",
        action="store_true",
        help="Skip rebuilding site/index.html static web application.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON_DAYS,
        help=f"Minimum forecast horizon in days from last observation (default: {DEFAULT_HORIZON_DAYS}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    payload = run_full_pipeline(
        refresh_web=bool(args.refresh_web),
        run_cv=not bool(args.skip_cv),
        rebuild_site=not bool(args.no_site),
        horizon_days=int(args.horizon),
    )
    win = payload["forecast_window"]
    cv = payload["cv_benchmark"]
    print(
        f"Banff Visit Planner updated: {win['start']} .. {win['end']} ({win['n_days']} days) | "
        f"OOS Cov80={cv['empirical_cov80_pct']}% Cov95={cv['empirical_cov95_pct']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
