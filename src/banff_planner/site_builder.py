"""Self-contained static web application builder (outputs site/index.html for GitHub Pages)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from banff_planner.config import SITE_DIR

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Banff Visit Planner — 15-Day Crowd &amp; Weather Forecast</title>
  <meta name="description" content="Data-driven 15-day trip planner for Banff, Alberta combining Town of Banff entrance traffic counters, weather forecasts, Alberta holiday bridges, and conformal uncertainty intervals." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #F6F7F5;
      --surface: #FFFFFF;
      --surface-alt: #EEF1ED;
      --surface-subtle: #F9FAF8;
      --border: #DCE2DC;
      --border-strong: #C5CEC6;
      --ink: #14201C;
      --muted: #52605A;
      --muted-soft: #75837D;
      --accent: #0F5C4D;
      --accent-ink: #FFFFFF;
      --accent-soft: rgba(15, 92, 77, 0.09);
      --glacier: #1D6FA5;
      --glacier-soft: rgba(29, 111, 165, 0.14);
      --glacier-softer: rgba(29, 111, 165, 0.07);
      --history: #64748B;
      --best: #15803D;
      --best-bg: #E8F6EE;
      --good: #3F7D20;
      --good-bg: #EEF7E6;
      --warn: #B45309;
      --warn-bg: #FDF3E1;
      --bad: #B91C1C;
      --bad-bg: #FDECEC;
      --weekend-band: rgba(180, 83, 9, 0.06);
      --shadow-sm: 0 1px 2px rgba(20, 32, 28, 0.04), 0 4px 12px rgba(20, 32, 28, 0.03);
      --shadow-md: 0 2px 6px rgba(20, 32, 28, 0.05), 0 12px 28px rgba(20, 32, 28, 0.06);
      --radius-lg: 18px;
      --radius-md: 12px;
      --radius-sm: 8px;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    html[data-theme="dark"] {
      --bg: #0E1412;
      --surface: #161E1B;
      --surface-alt: #1E2824;
      --surface-subtle: #1A2320;
      --border: #2A3732;
      --border-strong: #394A43;
      --ink: #F1F5F3;
      --muted: #9BA8A2;
      --muted-soft: #76857E;
      --accent: #34B395;
      --accent-ink: #07221B;
      --accent-soft: rgba(52, 179, 149, 0.14);
      --glacier: #56B4E9;
      --glacier-soft: rgba(86, 180, 233, 0.18);
      --glacier-softer: rgba(86, 180, 233, 0.08);
      --history: #94A3B8;
      --best: #4ADE80;
      --best-bg: rgba(34, 197, 94, 0.15);
      --good: #A3E635;
      --good-bg: rgba(163, 230, 53, 0.14);
      --warn: #FBBF24;
      --warn-bg: rgba(245, 158, 11, 0.15);
      --bad: #F87171;
      --bad-bg: rgba(239, 68, 68, 0.15);
      --weekend-band: rgba(251, 191, 36, 0.06);
      --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.28);
      --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.36);
    }

    *, *::before, *::after {
      box-sizing: border-box;
    }

    html, body {
      margin: 0;
      padding: 0;
      background-color: var(--bg);
      color: var(--ink);
      font-family: var(--font-sans);
      font-size: 15px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      overflow-x: hidden;
    }

    .tabular {
      font-variant-numeric: tabular-nums;
    }

    .container {
      width: 100%;
      max-width: 1240px;
      margin: 0 auto;
      padding: 1.25rem 1.25rem 3.5rem;
    }

    /* Top Header */
    .topbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.15rem;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .brand-mark {
      width: 40px;
      height: 40px;
      border-radius: 11px;
      background: var(--accent);
      color: var(--accent-ink);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .brand-title {
      margin: 0;
      font-size: 1.45rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      line-height: 1.15;
    }

    .brand-meta {
      margin: 0.15rem 0 0;
      font-size: 0.84rem;
      color: var(--muted);
    }

    .topbar-actions {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.65rem;
    }

    .nav-tabs {
      display: inline-flex;
      background: var(--surface-alt);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 3px;
      gap: 2px;
    }

    .nav-tab {
      border: none;
      background: transparent;
      color: var(--muted);
      font-family: inherit;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.44rem 0.9rem;
      border-radius: 999px;
      cursor: pointer;
      transition: background 0.16s ease, color 0.16s ease;
      white-space: nowrap;
    }

    .nav-tab[aria-selected="true"] {
      background: var(--surface);
      color: var(--ink);
      box-shadow: var(--shadow-sm);
    }

    .nav-tab:hover:not([aria-selected="true"]) {
      color: var(--ink);
    }

    .icon-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--ink);
      font-family: inherit;
      font-size: 0.84rem;
      font-weight: 600;
      padding: 0.46rem 0.8rem;
      border-radius: 999px;
      cursor: pointer;
      text-decoration: none;
      transition: border-color 0.15s ease, background 0.15s ease;
    }

    .icon-btn:hover {
      border-color: var(--border-strong);
      background: var(--surface-alt);
    }

    /* Filter Bar */
    .filter-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.9rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 0.85rem 1.05rem;
      margin-bottom: 1.15rem;
      box-shadow: var(--shadow-sm);
    }

    .filter-group {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.5rem;
    }

    .filter-label {
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-right: 0.25rem;
    }

    .seg-group {
      display: inline-flex;
      background: var(--surface-alt);
      border-radius: 10px;
      padding: 3px;
      gap: 2px;
      flex-wrap: wrap;
    }

    .seg-btn {
      border: none;
      background: transparent;
      color: var(--muted);
      font-family: inherit;
      font-size: 0.83rem;
      font-weight: 600;
      padding: 0.36rem 0.72rem;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.15s ease, color 0.15s ease;
      white-space: nowrap;
    }

    .seg-btn[aria-pressed="true"] {
      background: var(--accent);
      color: var(--accent-ink);
    }

    .seg-btn:hover:not([aria-pressed="true"]) {
      color: var(--ink);
    }

    /* Views */
    .view-panel[hidden] {
      display: none !important;
    }

    /* Hero Decision Row */
    .hero-grid {
      display: grid;
      grid-template-columns: 1.08fr 1fr 1.02fr;
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 1.15rem 1.2rem;
      box-shadow: var(--shadow-sm);
    }

    .hero-card {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 205px;
    }

    .eyebrow {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--muted);
      margin: 0 0 0.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }

    .hero-date {
      font-size: 1.72rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      line-height: 1.12;
      margin: 0 0 0.45rem;
    }

    .hero-weather-row {
      display: flex;
      align-items: center;
      gap: 0.55rem;
      font-size: 0.9rem;
      color: var(--ink);
      font-weight: 600;
      margin: 0.35rem 0;
    }

    .hero-sub {
      font-size: 0.87rem;
      color: var(--muted);
      margin: 0.35rem 0 0;
      line-height: 1.45;
    }

    .hero-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      margin-top: 0.85rem;
      padding-top: 0.7rem;
      border-top: 1px solid var(--border);
      font-size: 0.83rem;
    }

    .signal-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.22rem 0.62rem;
      border-radius: 999px;
      font-size: 0.77rem;
      font-weight: 700;
      white-space: nowrap;
    }

    .tone-best { background: var(--best-bg); color: var(--best); }
    .tone-good { background: var(--good-bg); color: var(--good); }
    .tone-warning { background: var(--warn-bg); color: var(--warn); }
    .tone-bad { background: var(--bad-bg); color: var(--bad); }

    .score-chip {
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.8rem;
      font-weight: 700;
      padding: 0.2rem 0.55rem;
      border-radius: 8px;
      background: var(--accent-soft);
      color: var(--accent);
    }

    .text-link-btn {
      border: none;
      background: transparent;
      color: var(--accent);
      font-family: inherit;
      font-size: 0.83rem;
      font-weight: 700;
      cursor: pointer;
      padding: 0;
    }

    .text-link-btn:hover {
      text-decoration: underline;
    }

    /* Metrics context row */
    .context-strip {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.85rem;
      margin-bottom: 1.25rem;
    }

    .stat-mini {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 0.85rem 1rem;
    }

    .stat-mini-label {
      font-size: 0.73rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 0.25rem;
    }

    .stat-mini-value {
      font-size: 1.05rem;
      font-weight: 700;
      margin-bottom: 0.15rem;
    }

    .stat-mini-note {
      font-size: 0.8rem;
      color: var(--muted);
      line-height: 1.35;
    }

    /* 15-Day Strip */
    .section-head {
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      justify-content: space-between;
      gap: 0.5rem;
      margin: 0 0 0.65rem;
    }

    .section-title {
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.015em;
      margin: 0;
    }

    .section-hint {
      font-size: 0.82rem;
      color: var(--muted);
    }

    .outlook-strip {
      display: grid;
      grid-template-columns: repeat(8, minmax(128px, 1fr));
      gap: 0.6rem;
      margin-bottom: 1.25rem;
    }

    .day-tile {
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      text-align: left;
      gap: 0.35rem;
      padding: 0.75rem 0.75rem 0.7rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      cursor: pointer;
      font-family: inherit;
      color: var(--ink);
      transition: transform 0.12s ease, border-color 0.14s ease, box-shadow 0.14s ease, opacity 0.14s ease;
    }

    .day-tile:hover {
      transform: translateY(-1px);
      border-color: var(--border-strong);
      box-shadow: var(--shadow-sm);
    }

    .day-tile[aria-pressed="true"] {
      border-color: var(--accent);
      background: var(--surface-subtle);
      box-shadow: 0 0 0 2px var(--accent-soft), var(--shadow-sm);
    }

    .day-tile.is-dimmed {
      opacity: 0.42;
    }

    .tile-top {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.25rem;
    }

    .tile-date {
      font-size: 0.83rem;
      font-weight: 700;
    }

    .tile-dow {
      font-size: 0.74rem;
      color: var(--muted);
      font-weight: 600;
    }

    .tile-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .dot-best { background: var(--best); }
    .dot-good { background: var(--good); }
    .dot-warning { background: var(--warn); }
    .dot-bad { background: var(--bad); }

    .tile-wx {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.8rem;
      font-weight: 600;
      width: 100%;
    }

    .tile-vol-row {
      width: 100%;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin-top: 0.15rem;
    }

    .tile-vol {
      font-size: 0.92rem;
      font-weight: 700;
    }

    .tile-score {
      font-size: 0.75rem;
      color: var(--muted);
      font-weight: 600;
    }

    .tile-event {
      width: 100%;
      font-size: 0.7rem;
      font-weight: 600;
      color: var(--accent);
      background: var(--accent-soft);
      border-radius: 6px;
      padding: 0.14rem 0.36rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Main Workspace Split: Chart Left + Selected-Day Inspector Right */
    .workspace-grid {
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 1.05rem;
      margin-bottom: 1.4rem;
      align-items: start;
    }

    .chart-card {
      padding: 1.15rem 1.2rem;
    }

    .chart-controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.6rem;
      margin-bottom: 0.75rem;
    }

    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
    }

    .toggle-chip {
      border: 1px solid var(--border);
      background: var(--surface-alt);
      color: var(--muted);
      font-family: inherit;
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.3rem 0.65rem;
      border-radius: 999px;
      cursor: pointer;
    }

    .toggle-chip[aria-pressed="true"] {
      background: var(--ink);
      color: var(--surface);
      border-color: var(--ink);
    }

    .canvas-wrap {
      position: relative;
      width: 100%;
      height: 360px;
    }

    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }

    .chart-tooltip {
      position: absolute;
      pointer-events: none;
      background: var(--surface);
      border: 1px solid var(--border-strong);
      border-radius: 10px;
      padding: 0.55rem 0.75rem;
      box-shadow: var(--shadow-md);
      font-size: 0.8rem;
      min-width: 195px;
      z-index: 10;
      opacity: 0;
      transition: opacity 0.1s ease;
    }

    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 0.9rem;
      margin-top: 0.75rem;
      padding-top: 0.65rem;
      border-top: 1px solid var(--border);
      font-size: 0.78rem;
      color: var(--muted);
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 0.38rem;
    }

    .legend-swatch {
      width: 14px;
      height: 4px;
      border-radius: 2px;
      display: inline-block;
    }

    /* Selected Day Inspector */
    .inspector-card {
      padding: 1.15rem 1.2rem;
    }

    .inspector-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 0.75rem;
      padding-bottom: 0.85rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 0.95rem;
    }

    .inspector-title {
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin: 0.15rem 0 0.25rem;
    }

    .inspector-metric-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.65rem;
      margin-bottom: 0.95rem;
    }

    .metric-box {
      background: var(--surface-alt);
      border-radius: var(--radius-md);
      padding: 0.7rem 0.85rem;
    }

    .metric-box-label {
      font-size: 0.73rem;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .metric-box-val {
      font-size: 1.15rem;
      font-weight: 800;
      margin: 0.15rem 0 0.08rem;
    }

    .metric-box-sub {
      font-size: 0.77rem;
      color: var(--muted);
    }

    .sub-heading {
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin: 0.9rem 0 0.45rem;
    }

    .pct-bar-wrap {
      margin-bottom: 0.55rem;
    }

    .pct-bar-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.81rem;
      margin-bottom: 0.24rem;
    }

    .pct-track {
      height: 7px;
      border-radius: 999px;
      background: var(--surface-alt);
      overflow: hidden;
    }

    .pct-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.2s ease;
    }

    /* Driver rows */
    .driver-list {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
    }

    .driver-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.5rem;
      align-items: center;
      font-size: 0.82rem;
      padding: 0.42rem 0.6rem;
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
    }

    .driver-impact {
      font-weight: 700;
      font-size: 0.8rem;
      padding: 0.1rem 0.42rem;
      border-radius: 6px;
    }

    .impact-up { background: var(--warn-bg); color: var(--warn); }
    .impact-down { background: var(--best-bg); color: var(--best); }

    /* Entrance split bar */
    .entrance-bar {
      display: flex;
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: var(--surface-alt);
      margin: 0.35rem 0 0.4rem;
    }

    .entrance-west { background: var(--glacier); }
    .entrance-east { background: var(--accent); }

    .entrance-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      color: var(--ink);
      gap: 0.5rem;
    }

    .tip-box {
      margin-top: 0.85rem;
      padding: 0.68rem 0.82rem;
      background: var(--accent-soft);
      border-radius: var(--radius-md);
      font-size: 0.82rem;
      color: var(--ink);
      line-height: 1.45;
    }

    /* Annual Calendar & Benchmark Views */
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .heatmap-wrap {
      overflow-x: auto;
      padding-bottom: 0.4rem;
    }

    .heatmap-grid {
      min-width: 780px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .hm-row {
      display: grid;
      grid-template-columns: 42px repeat(31, 1fr);
      gap: 3px;
      align-items: center;
    }

    .hm-month {
      font-size: 0.77rem;
      font-weight: 700;
      color: var(--muted);
    }

    .hm-cell {
      height: 22px;
      border-radius: 4px;
      border: 1px solid rgba(0, 0, 0, 0.04);
      cursor: pointer;
      transition: transform 0.1s ease;
    }

    .hm-cell:hover {
      transform: scale(1.18);
      z-index: 2;
      outline: 2px solid var(--ink);
    }

    .hm-cell.hm-empty {
      background: transparent;
      border: none;
      pointer-events: none;
    }

    .hm-readout {
      margin-top: 0.7rem;
      padding: 0.6rem 0.85rem;
      background: var(--surface-alt);
      border-radius: var(--radius-sm);
      font-size: 0.85rem;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 0.5rem;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
    }

    th, td {
      padding: 0.58rem 0.65rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }

    th {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      font-weight: 700;
    }

    tr.is-ensemble {
      background: var(--accent-soft);
      font-weight: 700;
    }

    .sources-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.85rem;
      margin-top: 1rem;
    }

    .source-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 0.9rem 1rem;
    }

    .source-title {
      font-weight: 700;
      font-size: 0.9rem;
      margin: 0 0 0.25rem;
    }

    .source-meta {
      font-size: 0.81rem;
      color: var(--muted);
      margin: 0;
      line-height: 1.4;
    }

    .footer {
      margin-top: 2rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 0.6rem;
      font-size: 0.8rem;
      color: var(--muted);
    }

    .footer a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }

    @media (max-width: 1080px) {
      .hero-grid { grid-template-columns: 1fr 1fr; }
      .hero-grid > :last-child { grid-column: 1 / -1; }
      .context-strip { grid-template-columns: repeat(2, 1fr); }
      .outlook-strip { grid-template-columns: repeat(4, 1fr); }
      .workspace-grid { grid-template-columns: 1fr; }
      .sources-grid { grid-template-columns: repeat(2, 1fr); }
    }

    @media (max-width: 720px) {
      .container { padding: 0.85rem 0.85rem 2.5rem; }
      .topbar-actions {
        width: 100%;
        min-width: 0;
      }
      .nav-tabs {
        width: 100%;
        max-width: 100%;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
      .nav-tab {
        min-width: 0;
        white-space: normal;
        text-align: center;
        line-height: 1.2;
        padding: 0.44rem 0.25rem;
        font-size: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .hero-grid { grid-template-columns: 1fr; }
      .context-strip { grid-template-columns: 1fr 1fr; }
      .outlook-strip {
        display: flex;
        overflow-x: auto;
        scroll-snap-type: x mandatory;
        padding-bottom: 0.45rem;
      }
      .day-tile {
        min-width: 146px;
        flex: 0 0 auto;
        scroll-snap-align: start;
      }
      .two-col, .sources-grid { grid-template-columns: 1fr; }
      .canvas-wrap { height: 290px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 19L9.5 7l4.5 7 3-4 4 9H3z"/>
          </svg>
        </div>
        <div>
          <h1 class="brand-title">Banff Visit Planner</h1>
          <p class="brand-meta" id="header-meta">Loading live forecast window…</p>
        </div>
      </div>

      <div class="topbar-actions">
        <div class="nav-tabs" role="tablist" aria-label="App Views">
          <button class="nav-tab" role="tab" id="tab-planner" aria-selected="true" data-view="planner">15-Day Planner</button>
          <button class="nav-tab" role="tab" id="tab-annual" aria-selected="false" data-view="annual">Year-Round Guide</button>
          <button class="nav-tab" role="tab" id="tab-science" aria-selected="false" data-view="science">Accuracy &amp; Sources</button>
        </div>
        <button class="icon-btn" id="theme-toggle" type="button" aria-label="Toggle color theme">
          <span id="theme-icon">☾</span>
          <span id="theme-label">Dark</span>
        </button>
        <a class="icon-btn" href="https://banff-visit-planner-app.streamlit.app/" target="_blank" rel="noopener">Streamlit Alt ↗</a>
      </div>
    </header>

    <!-- VIEW 1: 15-DAY TRIP PLANNER -->
    <section id="view-planner" class="view-panel">
      <div class="filter-bar" aria-label="Trip planning controls">
        <div class="filter-group">
          <span class="filter-label">Schedule</span>
          <div class="seg-group" id="ctrl-day-filter">
            <button class="seg-btn" data-filter="All upcoming days" aria-pressed="true" type="button">All days</button>
            <button class="seg-btn" data-filter="Weekdays only" aria-pressed="false" type="button">Weekdays</button>
            <button class="seg-btn" data-filter="Weekends only" aria-pressed="false" type="button">Weekends</button>
          </div>
        </div>

        <div class="filter-group">
          <span class="filter-label">Optimize for</span>
          <div class="seg-group" id="ctrl-priority">
            <button class="seg-btn" data-priority="balanced" aria-pressed="true" type="button">Balanced</button>
            <button class="seg-btn" data-priority="quiet" aria-pressed="false" type="button">Least Traffic</button>
            <button class="seg-btn" data-priority="weather" aria-pressed="false" type="button">Best Weather</button>
          </div>
        </div>

        <div class="filter-group">
          <span class="filter-label">Stay</span>
          <div class="seg-group" id="ctrl-trip-days">
            <button class="seg-btn" data-days="1" aria-pressed="false" type="button">1d</button>
            <button class="seg-btn" data-days="2" aria-pressed="false" type="button">2d</button>
            <button class="seg-btn" data-days="3" aria-pressed="true" type="button">3d</button>
            <button class="seg-btn" data-days="4" aria-pressed="false" type="button">4d</button>
          </div>
        </div>
      </div>

      <!-- Instant Decision Hero -->
      <div class="hero-grid" id="hero-grid"></div>

      <!-- Four Clear Statistical Context Cards -->
      <div class="context-strip" id="context-strip"></div>

      <!-- 15-Day Interactive Outlook Strip -->
      <div class="section-head">
        <h2 class="section-title">Upcoming Day-by-Day Outlook</h2>
        <span class="section-hint">Select any day to inspect traffic drivers, park entrances &amp; weather details</span>
      </div>
      <div class="outlook-strip" id="outlook-strip"></div>

      <!-- Interactive Canvas Chart + Selected Day Deep Dive -->
      <div class="workspace-grid">
        <div class="card chart-card">
          <div class="chart-controls">
            <div>
              <h2 class="section-title">Daily Traffic Trajectory &amp; Conformal Bounds</h2>
              <span class="section-hint">Verified two-way counts + log-scale multi-model ensemble</span>
            </div>
            <div class="chip-row">
              <button class="toggle-chip" id="tog-pi80" aria-pressed="true" type="button">80% Conformal</button>
              <button class="toggle-chip" id="tog-pi95" aria-pressed="false" type="button">95% Band</button>
              <button class="toggle-chip" id="tog-hist" aria-pressed="true" type="button">35-Day History</button>
              <button class="toggle-chip" id="tog-wknd" aria-pressed="true" type="button">Weekends</button>
            </div>
          </div>
          <div class="canvas-wrap">
            <canvas id="forecast-canvas"></canvas>
            <div class="chart-tooltip" id="chart-tooltip"></div>
          </div>
          <div class="chart-legend">
            <span class="legend-item"><span class="legend-swatch" style="background: var(--history);"></span>Observed history</span>
            <span class="legend-item"><span class="legend-swatch" style="background: var(--glacier);"></span>Ensemble forecast</span>
            <span class="legend-item"><span class="legend-swatch" style="background: var(--glacier-soft); height: 10px;"></span>80% Conformal PI</span>
            <span class="legend-item"><span class="legend-swatch" style="background: var(--best);"></span>Best pick</span>
            <span class="legend-item"><span class="legend-swatch" style="background: var(--bad);"></span>Peak avoid</span>
          </div>
        </div>

        <aside class="card inspector-card" id="day-inspector" aria-live="polite"></aside>
      </div>
    </section>

    <!-- VIEW 2: YEAR-ROUND GUIDE -->
    <section id="view-annual" class="view-panel" hidden>
      <div class="card" style="margin-bottom: 1rem;">
        <div class="section-head">
          <div>
            <h2 class="section-title">365-Day Banff Seasonal Crowd &amp; Weather Calendar</h2>
            <span class="section-hint">Trailing 3-year smoothed daily volume &amp; BANFF CS climate normals — hover or tap any calendar date</span>
          </div>
          <span class="signal-pill tone-best">May · Early June · Early Nov = Calmest</span>
        </div>
        <div class="heatmap-wrap">
          <div class="heatmap-grid" id="annual-heatmap"></div>
        </div>
        <div class="hm-readout" id="hm-readout">
          <span>Hover or tap any day in the calendar grid to inspect historical crowd volume and normal Banff temperatures.</span>
        </div>
      </div>

      <div class="two-col">
        <div class="card">
          <div class="section-head">
            <h3 class="section-title">Day-of-Week Traffic Rhythm</h3>
            <span class="section-hint" id="insight-weekly-text"></span>
          </div>
          <div class="canvas-wrap" style="height: 270px;">
            <canvas id="dow-canvas"></canvas>
          </div>
        </div>
        <div class="card">
          <div class="section-head">
            <h3 class="section-title">Monthly Seasonality (Past 3 Years)</h3>
            <span class="section-hint" id="insight-seasonal-text"></span>
          </div>
          <div class="canvas-wrap" style="height: 270px;">
            <canvas id="monthly-canvas"></canvas>
          </div>
        </div>
      </div>
    </section>

    <!-- VIEW 3: ACCURACY & DATA SOURCES -->
    <section id="view-science" class="view-panel" hidden>
      <div class="card" style="margin-bottom: 1rem;">
        <div class="section-head">
          <div>
            <h2 class="section-title">12-Fold Expanding Cross-Validation Benchmark (Unseen Out-of-Sample Days)</h2>
            <span class="section-hint" id="cv-subtitle"></span>
          </div>
          <span class="score-chip" id="cv-cov-badge"></span>
        </div>
        <div style="overflow-x: auto;">
          <table id="cv-table"></table>
        </div>
      </div>

      <div class="two-col">
        <div class="card">
          <h3 class="section-title" style="margin-bottom: 0.45rem;">Dynamic Log-Ensemble Composition</h3>
          <p class="section-hint" style="margin: 0 0 0.85rem;">All weights are learned automatically via inverse out-of-sample RMSE⁴ across orthogonal model families (zero identical clones).</p>
          <div id="weights-bars"></div>
        </div>
        <div class="card">
          <h3 class="section-title" style="margin-bottom: 0.45rem;">Methodology Fixes Over Prototype v1</h3>
          <ul style="margin: 0.3rem 0 0; padding-left: 1.15rem; font-size: 0.86rem; color: var(--muted); line-height: 1.55;">
            <li><strong>Multiplicative ln(y) modeling:</strong> Matches Banff's summer/winter variance scaling.</li>
            <li><strong>Zero duplicate models:</strong> Replaces cloned NBEATSx checkpoints with CatBoost residual, MLForecast LGBM, Ridge+AR(1), and MSTL+ETS.</li>
            <li><strong>Real Conformal Intervals (80%/95%):</strong> Stratified by weekday vs weekend &amp; horizon instead of heuristic min/max model range.</li>
            <li><strong>Conditional Day-Type Percentiles &amp; Contiguous Windows:</strong> Compares weekends fairly to seasonal weekends and validates contiguous calendar spans.</li>
          </ul>
        </div>
      </div>

      <div class="sources-grid" id="sources-grid"></div>
    </section>

    <footer class="footer">
      <span>Banff Visit Planner v2.0 · Traffic counts via Town of Banff Open Tableau · Climate via ECCC BANFF CS &amp; Open-Meteo · Macro via BoC, StatCan &amp; Alberta Economic Dashboard</span>
      <span><a href="https://github.com/v-datos/Banff_Visit_Planner_App" target="_blank" rel="noopener">GitHub Repository</a> · Daily automated CI refresh</span>
    </footer>
  </div>

  <script id="banff-payload-data" type="application/json">__PAYLOAD_JSON__</script>
  <script id="banff-planner-logic">
    (function (root) {
      "use strict";

      function filterForecastDays(days, dayFilter) {
        var upcoming = (days || []).filter(function (d) { return !d.is_bridge_gap; });
        var list = upcoming.length ? upcoming : (days || []).slice();
        if (dayFilter === "Weekdays only") {
          return list.filter(function (d) { return !d.is_weekend; }).sort(byDs);
        }
        if (dayFilter === "Weekends only") {
          return list.filter(function (d) { return !!d.is_weekend; }).sort(byDs);
        }
        return list.slice().sort(byDs);
      }

      function byDs(a, b) {
        return a.ds < b.ds ? -1 : a.ds > b.ds ? 1 : 0;
      }

      function getScoreKey(dayFilter, priority) {
        var isScoped = dayFilter === "Weekdays only" || dayFilter === "Weekends only";
        if (priority === "quiet") {
          return isScoped ? "traffic_score_daytype" : "traffic_score_overall";
        }
        if (priority === "weather") {
          return "visit_score_weather";
        }
        return isScoped ? "visit_score_scoped_balanced" : "visit_score_balanced";
      }

      function daysDiff(dsA, dsB) {
        var a = new Date(dsA + "T00:00:00Z").getTime();
        var b = new Date(dsB + "T00:00:00Z").getTime();
        return Math.round((b - a) / 86400000);
      }

      function computeBestWindow(filteredDays, dayFilter, tripDays, scoreKey) {
        var list = (filteredDays || []).slice().sort(byDs);
        if (!list.length) {
          return { text: "No matching days", start: null, end: null, actual_days: 0, avg_vehicles: null, avg_score: 0 };
        }
        var reqK = Math.max(1, Math.min(Number(tripDays) || 3, 4));
        if (dayFilter === "Weekends only" && reqK > 2) {
          reqK = 2;
        }
        var best = null;
        for (var k = reqK; k >= 1; k--) {
          var candidates = [];
          for (var i = 0; i <= list.length - k; i++) {
            var chunk = list.slice(i, i + k);
            if (daysDiff(chunk[0].ds, chunk[chunk.length - 1].ds) === k - 1) {
              var sumVeh = 0;
              var sumSc = 0;
              for (var j = 0; j < chunk.length; j++) {
                sumVeh += Number(chunk[j].ensemble || 0);
                sumSc += Number(chunk[j][scoreKey] || 0);
              }
              candidates.push({
                start: chunk[0].ds,
                end: chunk[chunk.length - 1].ds,
                startShort: chunk[0].dow_short + ", " + chunk[0].date_short,
                endShort: chunk[chunk.length - 1].dow_short + ", " + chunk[chunk.length - 1].date_short,
                actual_days: k,
                avg_vehicles: Math.round(sumVeh / k),
                avg_score: Math.round((sumSc / k) * 10) / 10
              });
            }
          }
          if (candidates.length) {
            candidates.sort(function (a, b) {
              if (scoreKey.indexOf("traffic_score") === 0) {
                if (a.avg_vehicles !== b.avg_vehicles) return a.avg_vehicles - b.avg_vehicles;
                return b.avg_score - a.avg_score;
              }
              if (b.avg_score !== a.avg_score) return b.avg_score - a.avg_score;
              return a.avg_vehicles - b.avg_vehicles;
            });
            best = candidates[0];
            break;
          }
        }
        if (!best) {
          var sorted = list.slice().sort(function (a, b) { return a.ensemble - b.ensemble; });
          var d0 = sorted[0];
          return {
            text: d0.dow_short + ", " + d0.date_short,
            start: d0.ds,
            end: d0.ds,
            actual_days: 1,
            avg_vehicles: Math.round(d0.ensemble),
            avg_score: Number(d0[scoreKey] || 0)
          };
        }
        return {
          text: best.actual_days === 1 ? best.startShort : best.startShort + " to " + best.endShort,
          start: best.start,
          end: best.end,
          actual_days: best.actual_days,
          avg_vehicles: best.avg_vehicles,
          avg_score: best.avg_score
        };
      }

      function computeUnbiasedTrend(recentHistory, filteredDays, dayFilter) {
        if (!filteredDays.length || !(recentHistory || []).length) {
          return { label: "Steady trend", detail: "Matching seasonal conditions." };
        }
        var pool = recentHistory.filter(function (r) {
          if (dayFilter === "Weekdays only") return !r.is_weekend;
          if (dayFilter === "Weekends only") return !!r.is_weekend;
          return true;
        });
        var tail = pool.slice(-30);
        if (!tail.length) tail = recentHistory.slice(-30);
        var histMean = tail.reduce(function (s, r) { return s + Number(r.vehicles || 0); }, 0) / tail.length;
        var nextSubset = filteredDays.slice(0, Math.min(7, filteredDays.length));
        var nextMean = nextSubset.reduce(function (s, d) { return s + Number(d.ensemble || 0); }, 0) / nextSubset.length;
        var normMean = nextSubset.reduce(function (s, d) { return s + Number(d.dow_seasonal_norm || 0); }, 0) / nextSubset.length;
        var normNote = normMean > 0
          ? " (" + (nextMean >= normMean ? "+" : "") + Math.round(((nextMean - normMean) / normMean) * 100) + "% vs 3-yr norm)"
          : "";
        var scopeTxt = dayFilter === "Weekdays only" ? "recent weekdays" : dayFilter === "Weekends only" ? "recent weekends" : "recent 30 days";
        var delta = histMean > 0 ? ((nextMean - histMean) / histMean) * 100 : 0;
        if (delta <= -6) {
          return { label: "Quieter trend", detail: Math.round(Math.abs(delta)) + "% below " + scopeTxt + normNote + "." };
        }
        if (delta >= 6) {
          return { label: "Busier trend", detail: Math.round(delta) + "% above " + scopeTxt + normNote + "." };
        }
        return { label: "Steady trend", detail: "Within " + Math.round(Math.abs(delta)) + "% of " + scopeTxt + normNote + "." };
      }

      function computeRecommendations(payload, dayFilter, priority, tripDays) {
        var filtered = filterForecastDays(payload.forecast_days, dayFilter);
        if (!filtered.length) return null;
        var isScoped = dayFilter === "Weekdays only" || dayFilter === "Weekends only";
        var scoreKey = getScoreKey(dayFilter, priority);
        var pctKey = isScoped ? "daytype_percentile" : "overall_percentile";

        var ranked = filtered.slice().sort(function (a, b) {
          if (Number(b[scoreKey] || 0) !== Number(a[scoreKey] || 0)) {
            return Number(b[scoreKey] || 0) - Number(a[scoreKey] || 0);
          }
          return Number(a.ensemble || 0) - Number(b.ensemble || 0);
        });
        var busiest = filtered.slice().sort(function (a, b) {
          return Number(b.ensemble || 0) - Number(a.ensemble || 0);
        });
        var busyCount = filtered.filter(function (d) { return Number(d[pctKey] || 0) > 80; }).length;
        return {
          filtered: filtered,
          isScoped: isScoped,
          scoreKey: scoreKey,
          pctKey: pctKey,
          bestDay: ranked[0],
          worstDay: busiest[0],
          bestWindow: computeBestWindow(filtered, dayFilter, tripDays, scoreKey),
          trend: computeUnbiasedTrend(payload.recent_history, filtered, dayFilter),
          busyCount: busyCount
        };
      }

      root.BanffPlannerLogic = {
        filterForecastDays: filterForecastDays,
        computeBestWindow: computeBestWindow,
        computeUnbiasedTrend: computeUnbiasedTrend,
        computeRecommendations: computeRecommendations,
        getScoreKey: getScoreKey
      };
    })(typeof globalThis !== "undefined" ? globalThis : this);
  </script>
  <script>
    (function () {
      "use strict";
      if (typeof document === "undefined") return;

      var rawEl = document.getElementById("banff-payload-data");
      var PAYLOAD = rawEl ? JSON.parse(rawEl.textContent) : null;
      if (!PAYLOAD) return;

      var L = window.BanffPlannerLogic;
      var upcomingDays = (PAYLOAD.forecast_days || []).filter(function (d) { return !d.is_bridge_gap; });
      if (!upcomingDays.length) upcomingDays = PAYLOAD.forecast_days || [];

      var state = {
        view: "planner",
        dayFilter: "All upcoming days",
        priority: "balanced",
        tripDays: 3,
        selectedDs: upcomingDays[0] ? upcomingDays[0].ds : null,
        showPi80: true,
        showPi95: false,
        showHist: true,
        showWknd: true
      };

      // Initialize default selectedDs to initial best day
      var initRec = L.computeRecommendations(PAYLOAD, state.dayFilter, state.priority, state.tripDays);
      if (initRec && initRec.bestDay) state.selectedDs = initRec.bestDay.ds;

      // Theme persistence + optional ?theme=dark|light & ?view=planner|annual|science
      var urlParams = typeof URLSearchParams !== "undefined" ? new URLSearchParams(window.location.search) : null;
      var qTheme = urlParams ? urlParams.get("theme") : null;
      var qView = urlParams ? urlParams.get("view") : null;
      var hashView = (window.location.hash || "").replace("#", "");
      var savedTheme = null;
      try { savedTheme = localStorage.getItem("banff_planner_theme"); } catch (e) {}
      var initTheme = (qTheme === "dark" || qTheme === "light") ? qTheme : savedTheme;
      if (initTheme === "dark" || initTheme === "light") {
        document.documentElement.setAttribute("data-theme", initTheme);
      }
      updateThemeBtn();

      function updateThemeBtn() {
        var cur = document.documentElement.getAttribute("data-theme") || "light";
        document.getElementById("theme-icon").textContent = cur === "dark" ? "☀" : "☾";
        document.getElementById("theme-label").textContent = cur === "dark" ? "Light" : "Dark";
      }

      document.getElementById("theme-toggle").addEventListener("click", function () {
        var cur = document.documentElement.getAttribute("data-theme") || "light";
        var next = cur === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        try { localStorage.setItem("banff_planner_theme", next); } catch (e) {}
        updateThemeBtn();
        drawChartsForActiveView();
      });

      function cssVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      }

      function fmtNum(v) {
        if (v === null || v === undefined || isNaN(v)) return "—";
        return Math.round(Number(v)).toLocaleString("en-US");
      }

      function fmtTemp(v) {
        if (v === null || v === undefined || isNaN(v)) return "—";
        return Math.round(Number(v)) + "°C";
      }

      function weatherSvg(iconKey, size) {
        var s = size || 20;
        var base = 'width="' + s + '" height="' + s + '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';
        if (iconKey === "sun") {
          return '<svg ' + base + ' style="color:#D97706"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32l1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41m11.32-11.32l1.41-1.41"/></svg>';
        }
        if (iconKey === "sun-cloud" || iconKey === "partly-cloudy") {
          return '<svg ' + base + ' style="color:#D97706"><path d="M12 3v2M5.6 5.6l1.4 1.4M3 12h2m11-4a4 4 0 0 0-7.5-1.5"/><path stroke="currentColor" style="color:var(--muted)" d="M7.5 19h9a4 4 0 0 0 .5-7.97A5.5 5.5 0 0 0 6.2 12.6 3.5 3.5 0 0 0 7.5 19z"/></svg>';
        }
        if (iconKey === "snow" || iconKey === "snow-heavy") {
          return '<svg ' + base + ' style="color:var(--glacier)"><path d="M7 15h10a4 4 0 0 0 .5-7.97A5.5 5.5 0 0 0 6.5 8.5 3.5 3.5 0 0 0 7 15z"/><path d="M8 18v2m4-2v2m4-2v2"/></svg>';
        }
        if (iconKey === "rain" || iconKey === "rain-heavy" || iconKey === "drizzle" || iconKey === "sleet") {
          return '<svg ' + base + ' style="color:var(--glacier)"><path d="M7 14h10a4 4 0 0 0 .5-7.97A5.5 5.5 0 0 0 6.5 7.5 3.5 3.5 0 0 0 7 14z"/><path d="M8.5 17.5l-1 3m4.5-3l-1 3m4.5-3l-1 3"/></svg>';
        }
        if (iconKey === "storm") {
          return '<svg ' + base + ' style="color:var(--warn)"><path d="M7 14h10a4 4 0 0 0 .5-7.97A5.5 5.5 0 0 0 6.5 7.5 3.5 3.5 0 0 0 7 14z"/><path d="M13 14l-2 4h3l-2 4"/></svg>';
        }
        return '<svg ' + base + ' style="color:var(--muted)"><path d="M7 17h10a4 4 0 0 0 .5-7.97A5.5 5.5 0 0 0 6.5 10.5 3.5 3.5 0 0 0 7 17z"/></svg>';
      }

      // Header Metadata
      var fresh = PAYLOAD.data_freshness || {};
      document.getElementById("header-meta").textContent =
        "Updated " + (fresh.generated_date_mt || PAYLOAD.generated_date_mt) +
        " · Forecast covers " + upcomingDays.length + " upcoming days (" +
        upcomingDays[0].date_short + " – " + upcomingDays[upcomingDays.length - 1].date_short + ")" +
        " · Traffic data through " + (fresh.traffic_last_observed || "latest");

      function setActiveView(viewName) {
        if (viewName !== "planner" && viewName !== "annual" && viewName !== "science") viewName = "planner";
        state.view = viewName;
        document.querySelectorAll(".nav-tab").forEach(function (b) {
          b.setAttribute("aria-selected", b.getAttribute("data-view") === state.view ? "true" : "false");
        });
        document.querySelectorAll(".view-panel").forEach(function (p) {
          p.hidden = p.id !== "view-" + state.view;
        });
        drawChartsForActiveView();
      }

      // Tab Switching
      document.querySelectorAll(".nav-tab").forEach(function (btn) {
        btn.addEventListener("click", function () {
          setActiveView(btn.getAttribute("data-view"));
        });
      });

      // Control Bar Handlers
      document.querySelectorAll("#ctrl-day-filter .seg-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          state.dayFilter = btn.getAttribute("data-filter");
          document.querySelectorAll("#ctrl-day-filter .seg-btn").forEach(function (b) {
            b.setAttribute("aria-pressed", b === btn ? "true" : "false");
          });
          var r = L.computeRecommendations(PAYLOAD, state.dayFilter, state.priority, state.tripDays);
          if (r && r.bestDay && r.filtered.every(function (d) { return d.ds !== state.selectedDs; })) {
            state.selectedDs = r.bestDay.ds;
          }
          renderPlanner();
        });
      });

      document.querySelectorAll("#ctrl-priority .seg-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          state.priority = btn.getAttribute("data-priority");
          document.querySelectorAll("#ctrl-priority .seg-btn").forEach(function (b) {
            b.setAttribute("aria-pressed", b === btn ? "true" : "false");
          });
          var r = L.computeRecommendations(PAYLOAD, state.dayFilter, state.priority, state.tripDays);
          if (r && r.bestDay) state.selectedDs = r.bestDay.ds;
          renderPlanner();
        });
      });

      document.querySelectorAll("#ctrl-trip-days .seg-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          state.tripDays = Number(btn.getAttribute("data-days"));
          document.querySelectorAll("#ctrl-trip-days .seg-btn").forEach(function (b) {
            b.setAttribute("aria-pressed", b === btn ? "true" : "false");
          });
          renderPlanner();
        });
      });

      function bindToggle(id, prop) {
        var el = document.getElementById(id);
        el.addEventListener("click", function () {
          state[prop] = !state[prop];
          el.setAttribute("aria-pressed", state[prop] ? "true" : "false");
          drawForecastChart();
        });
      }
      bindToggle("tog-pi80", "showPi80");
      bindToggle("tog-pi95", "showPi95");
      bindToggle("tog-hist", "showHist");
      bindToggle("tog-wknd", "showWknd");

      function renderPlanner() {
        var rec = L.computeRecommendations(PAYLOAD, state.dayFilter, state.priority, state.tripDays);
        if (!rec) return;

        var best = rec.bestDay;
        var worst = rec.worstDay;
        var win = rec.bestWindow;
        var bestSignal = rec.isScoped ? best.signal_daytype : best.signal_overall;
        var bestTone = rec.isScoped ? best.tone_daytype : best.tone_overall;
        var worstSignal = rec.isScoped ? worst.signal_daytype : worst.signal_overall;
        var worstTone = rec.isScoped ? worst.tone_daytype : worst.tone_overall;
        var bestScore = Math.round(Number(best[rec.scoreKey] || 0));

        var bestDrivers = (best.driver_attributions && best.driver_attributions.drivers) ? best.driver_attributions.drivers : [];
        var calmDriver = bestDrivers.filter(function (dr) { return Number(dr.impact_pct) < 0; })[0];
        var bestDriverTxt = calmDriver
          ? calmDriver.factor + " (" + calmDriver.impact_pct.toFixed(1) + "%)"
          : (bestDrivers[0] ? bestDrivers[0].factor : "Calm midweek traffic flow & comfortable conditions");

        var worstTip = worst.is_weekend
          ? "Park at Banff Train Station &amp; ride Roam Transit before 09:00 AM."
          : "Plan hikes before 09:30 AM or explore Bow Valley Parkway / Lake Minnewanka early.";

        document.getElementById("hero-grid").innerHTML =
          '<article class="card hero-card">' +
            '<div>' +
              '<div class="eyebrow"><span>Recommended Day</span><span class="score-chip tabular">Score ' + bestScore + '/100</span></div>' +
              '<h2 class="hero-date">' + best.dow_short + ", " + best.date_short + '</h2>' +
              '<span class="signal-pill tone-' + bestTone + '">' + bestSignal + '</span>' +
              '<div class="hero-weather-row">' + weatherSvg(best.weather_icon, 20) +
                '<span class="tabular">' + fmtTemp(best.temp_max_c) + ' / ' + fmtTemp(best.temp_min_c) + ' · ' + best.weather_label + '</span></div>' +
              '<p class="hero-sub"><strong class="tabular">' + fmtNum(best.ensemble) + ' vehicles/day</strong> (80% CI: ' +
                fmtNum(best.pi80_low) + '–' + fmtNum(best.pi80_high) + '). Primary driver: ' + bestDriverTxt + '.</p>' +
            '</div>' +
            '<div class="hero-footer">' +
              '<span class="tabular" style="color:var(--muted);">' + (best.vs_dow_norm_pct >= 0 ? "+" : "") + best.vs_dow_norm_pct + '% vs typical ' + best.dow_short + '</span>' +
              '<button class="text-link-btn" type="button" data-select-ds="' + best.ds + '">Inspect breakdown →</button>' +
            '</div>' +
          '</article>' +

          '<article class="card hero-card">' +
            '<div>' +
              '<div class="eyebrow"><span>Busiest Day to Avoid</span>' +
                (worst.event_label ? '<span class="signal-pill tone-warning">' + worst.event_label + '</span>' : '<span class="tabular" style="color:var(--muted);">Peak load</span>') +
              '</div>' +
              '<h2 class="hero-date">' + worst.dow_short + ", " + worst.date_short + '</h2>' +
              '<span class="signal-pill tone-' + worstTone + '">' + worstSignal + '</span>' +
              '<div class="hero-weather-row">' + weatherSvg(worst.weather_icon, 20) +
                '<span class="tabular">' + fmtTemp(worst.temp_max_c) + ' / ' + fmtTemp(worst.temp_min_c) + ' · ' + worst.weather_label + '</span></div>' +
              '<p class="hero-sub"><strong class="tabular">' + fmtNum(worst.ensemble) + ' vehicles/day</strong> (' +
                (worst.vs_dow_norm_pct >= 0 ? "+" : "") + worst.vs_dow_norm_pct + '% vs typical ' + worst.dow_short + '). ' + worstTip + '</p>' +
            '</div>' +
            '<div class="hero-footer">' +
              '<span class="tabular" style="color:var(--muted);">West ' + fmtNum(worst.west_entrance) + ' · East ' + fmtNum(worst.east_entrance) + '</span>' +
              '<button class="text-link-btn" type="button" data-select-ds="' + worst.ds + '">Inspect peak →</button>' +
            '</div>' +
          '</article>' +

          '<article class="card hero-card">' +
            '<div>' +
              '<div class="eyebrow"><span>Best ' + win.actual_days + '-Day Contiguous Stay</span><span class="score-chip tabular">Avg ' + fmtNum(win.avg_vehicles) + '/day</span></div>' +
              '<h2 class="hero-date" style="font-size:1.38rem;">' + win.text + '</h2>' +
              '<p class="hero-sub" style="margin-top:0.45rem;">' + rec.trend.label + ': ' + rec.trend.detail + '</p>' +
              '<p class="hero-sub">' + (PAYLOAD.insights && PAYLOAD.insights.conformal_detail ? PAYLOAD.insights.conformal_detail : "") + '</p>' +
            '</div>' +
            '<div class="hero-footer">' +
              '<span>' + rec.busyCount + ' heavy-crowd days in scope</span>' +
              '<button class="text-link-btn" type="button" data-select-ds="' + (win.start || best.ds) + '">Start of window →</button>' +
            '</div>' +
          '</article>';

        document.querySelectorAll("[data-select-ds]").forEach(function (el) {
          el.addEventListener("click", function () {
            state.selectedDs = el.getAttribute("data-select-ds");
            renderStrip();
            renderInspector();
            drawForecastChart();
          });
        });

        var scopeWord = rec.isScoped ? state.dayFilter.toLowerCase() : "all upcoming days";
        var bench = PAYLOAD.cv_benchmark || {};
        var ensRow = (bench.summary_metrics || []).filter(function (m) { return m.model === "Ensemble_Weighted"; })[0];
        document.getElementById("context-strip").innerHTML =
          '<div class="stat-mini"><div class="stat-mini-label">Multi-Day Window</div><div class="stat-mini-value tabular">' + win.text + '</div><div class="stat-mini-note">Strictly contiguous ' + win.actual_days + '-day calendar span</div></div>' +
          '<div class="stat-mini"><div class="stat-mini-label">Day-Type Trend</div><div class="stat-mini-value">' + rec.trend.label + '</div><div class="stat-mini-note">' + rec.trend.detail + '</div></div>' +
          '<div class="stat-mini"><div class="stat-mini-label">Backtest Precision</div><div class="stat-mini-value tabular">' + (ensRow ? ensRow.MAPE.toFixed(2) + "% MAPE" : "Calibrated") + '</div><div class="stat-mini-note">' + (bench.n_oof_days || 192) + ' unseen CV days · Cov80 ' + (bench.empirical_cov80_pct || 80) + '%</div></div>' +
          '<div class="stat-mini"><div class="stat-mini-label">Filtered Crowds</div><div class="stat-mini-value tabular">' + rec.busyCount + ' of ' + rec.filtered.length + ' days busy</div><div class="stat-mini-note">Evaluated fairly across ' + scopeWord + '</div></div>';

        renderStrip();
        renderInspector();
        drawForecastChart();
      }

      function renderStrip() {
        var rec = L.computeRecommendations(PAYLOAD, state.dayFilter, state.priority, state.tripDays);
        var matchSet = {};
        rec.filtered.forEach(function (d) { matchSet[d.ds] = true; });
        var html = upcomingDays.map(function (d) {
          var isMatch = !!matchSet[d.ds];
          var isSel = d.ds === state.selectedDs;
          var tone = rec.isScoped ? d.tone_daytype : d.tone_overall;
          var sc = Math.round(Number(d[rec.scoreKey] || 0));
          var precipNote = d.snowfall_cm >= 1.0
            ? d.snowfall_cm.toFixed(1) + "cm"
            : (d.precip_prob_pct !== null && d.precip_prob_pct >= 25 ? d.precip_prob_pct + "%" : "");
          return (
            '<button type="button" class="day-tile' + (isMatch ? "" : " is-dimmed") + '" aria-pressed="' + (isSel ? "true" : "false") + '" data-ds="' + d.ds + '">' +
              '<div class="tile-top">' +
                '<div><div class="tile-date">' + d.date_short + '</div><div class="tile-dow">' + d.dow_short + '</div></div>' +
                '<span class="tile-dot dot-' + tone + '" title="' + (rec.isScoped ? d.signal_daytype : d.signal_overall) + '"></span>' +
              '</div>' +
              '<div class="tile-wx">' + weatherSvg(d.weather_icon, 18) +
                '<span class="tabular">' + fmtTemp(d.temp_max_c) + '</span>' +
                (precipNote ? '<span class="tabular" style="color:var(--glacier);font-size:0.73rem;margin-left:auto;">' + precipNote + '</span>' : '') +
              '</div>' +
              '<div class="tile-vol-row">' +
                '<span class="tile-vol tabular">' + (d.ensemble / 1000).toFixed(1) + 'k</span>' +
                '<span class="tile-score tabular">' + sc + '/100</span>' +
              '</div>' +
              (d.event_label ? '<div class="tile-event" title="' + d.event_label + '">' + d.event_label + '</div>' : '') +
            '</button>'
          );
        }).join("");
        var stripEl = document.getElementById("outlook-strip");
        stripEl.innerHTML = html;
        stripEl.querySelectorAll(".day-tile").forEach(function (tile) {
          tile.addEventListener("click", function () {
            state.selectedDs = tile.getAttribute("data-ds");
            renderStrip();
            renderInspector();
            drawForecastChart();
          });
        });
      }

      function renderInspector() {
        var d = upcomingDays.filter(function (x) { return x.ds === state.selectedDs; })[0] || upcomingDays[0];
        if (!d) return;
        var rec = L.computeRecommendations(PAYLOAD, state.dayFilter, state.priority, state.tripDays);
        var sc = Math.round(Number(d[rec.scoreKey] || 0));
        var toneColor = d.tone_daytype === "best" ? "var(--best)" : d.tone_daytype === "good" ? "var(--good)" : d.tone_daytype === "warning" ? "var(--warn)" : "var(--bad)";
        var overallColor = d.tone_overall === "best" ? "var(--best)" : d.tone_overall === "good" ? "var(--good)" : d.tone_overall === "warning" ? "var(--warn)" : "var(--bad)";

        var drivers = (d.driver_attributions && d.driver_attributions.drivers && d.driver_attributions.drivers.length)
          ? d.driver_attributions.drivers
          : [{ factor: "Baseline seasonal & weekly rhythm", impact_pct: 0.0, category: "calendar" }];

        var driversHtml = drivers.map(function (dr) {
          var cls = dr.impact_pct >= 0 ? "impact-up" : "impact-down";
          var sign = dr.impact_pct >= 0 ? "+" : "";
          return '<div class="driver-row"><span>' + dr.factor + '</span><span class="driver-impact tabular ' + cls + '">' + sign + dr.impact_pct.toFixed(1) + '%</span></div>';
        }).join("");

        var wShare = Number(d.west_share_pct || 54);
        var eShare = Math.round((100 - wShare) * 10) / 10;
        var tip = "";
        if (d.snowfall_cm >= 2 || (d.temp_max_c !== null && d.temp_max_c <= -10)) {
          tip = "Winter alpine conditions (" + fmtTemp(d.temp_max_c) + (d.snowfall_cm > 0 ? ", " + d.snowfall_cm.toFixed(1) + " cm snow" : "") + "). Winter tires required on Hwy 93/Icefields Pkwy; layer for sub-zero valley winds.";
        } else if (d.ensemble >= 19000) {
          tip = "High park volume expected. Arrive before 08:45 AM, park at Banff Train Station, and use Roam Transit for Lake Louise / Sulphur Mountain.";
        } else if (d.daytype_percentile <= 35) {
          tip = "Smooth traffic flow window relative to typical " + d.dow + "s. Ideal for Banff Ave parking, Gondola arrival, or Bow Valley Parkway drives.";
        } else {
          tip = "Moderate visitor volume. Peak corridor congestion typically clusters between 10:30 AM and 3:30 PM on Banff Ave &amp; Bow River Bridge.";
        }

        document.getElementById("day-inspector").innerHTML =
          '<div class="inspector-header">' +
            '<div>' +
              '<div class="eyebrow" style="margin-bottom:0.2rem;">Selected Day Deep Dive</div>' +
              '<h3 class="inspector-title">' + d.dow + ', ' + d.date_short + '</h3>' +
              '<span class="signal-pill tone-' + (rec.isScoped ? d.tone_daytype : d.tone_overall) + '">' +
                (rec.isScoped ? d.signal_daytype : d.signal_overall) + '</span>' +
              (d.event_label ? ' <span class="signal-pill tone-warning" style="margin-left:0.25rem;">' + d.event_label + '</span>' : '') +
            '</div>' +
            '<span class="score-chip tabular">Visit ' + sc + '/100</span>' +
          '</div>' +

          '<div class="inspector-metric-row">' +
            '<div class="metric-box">' +
              '<div class="metric-box-label">Predicted Traffic</div>' +
              '<div class="metric-box-val tabular">' + fmtNum(d.ensemble) + ' <span style="font-size:0.78rem;font-weight:600;color:var(--muted);">veh</span></div>' +
              '<div class="metric-box-sub tabular">80% CI: ' + fmtNum(d.pi80_low) + '–' + fmtNum(d.pi80_high) + '</div>' +
            '</div>' +
            '<div class="metric-box">' +
              '<div class="metric-box-label">Banff Weather</div>' +
              '<div class="metric-box-val tabular" style="display:flex;align-items:center;gap:0.35rem;">' + weatherSvg(d.weather_icon, 19) + fmtTemp(d.temp_max_c) + ' <span style="font-size:0.82rem;color:var(--muted);font-weight:500;">/ ' + fmtTemp(d.temp_min_c) + '</span></div>' +
              '<div class="metric-box-sub">' + d.weather_label + (d.precip_mm > 0.1 ? ' · ' + d.precip_mm.toFixed(1) + 'mm' : '') + ' · Comfort ' + Math.round(d.weather_score) + '/100</div>' +
            '</div>' +
          '</div>' +

          '<div class="sub-heading">Fair Crowd Benchmarks</div>' +
          '<div class="pct-bar-wrap">' +
            '<div class="pct-bar-labels"><span>vs Same Day-of-Week (' + d.dow + 's, 3-yr norm ' + fmtNum(d.dow_seasonal_norm) + ')</span>' +
              '<strong class="tabular">' + Math.round(d.daytype_percentile) + 'th pct (' + (d.vs_dow_norm_pct >= 0 ? "+" : "") + d.vs_dow_norm_pct + '%)</strong></div>' +
            '<div class="pct-track"><div class="pct-fill" style="width:' + Math.min(100, Math.max(4, d.daytype_percentile)) + '%;background:' + toneColor + ';"></div></div>' +
          '</div>' +
          '<div class="pct-bar-wrap">' +
            '<div class="pct-bar-labels"><span>vs All Days in Same Season (±10d)</span><strong class="tabular">' + Math.round(d.overall_percentile) + 'th pct</strong></div>' +
            '<div class="pct-track"><div class="pct-fill" style="width:' + Math.min(100, Math.max(4, d.overall_percentile)) + '%;background:' + overallColor + ';"></div></div>' +
          '</div>' +

          '<div class="sub-heading">Why This Forecast?</div>' +
          '<div class="driver-list">' + driversHtml + '</div>' +

          '<div class="sub-heading">Expected Entrance &amp; Corridor Load</div>' +
          '<div class="entrance-bar">' +
            '<div class="entrance-west" style="width:' + wShare + '%;"></div>' +
            '<div class="entrance-east" style="width:' + eShare + '%;"></div>' +
          '</div>' +
          '<div class="entrance-labels tabular">' +
            '<span><strong style="color:var(--glacier);">West (Norquay / Bow Vly)</strong> ' + wShare.toFixed(0) + '% · ' + fmtNum(d.west_entrance) + '</span>' +
            '<span><strong style="color:var(--accent);">East (Banff Ave / Hwy 1)</strong> ' + eShare.toFixed(0) + '% · ' + fmtNum(d.east_entrance) + '</span>' +
          '</div>' +
          '<div style="font-size:0.78rem;color:var(--muted);margin-top:0.25rem;" class="tabular">Mountain Ave / Sulphur Mountain Gondola corridor: ~' + fmtNum(d.mountain_ave) + ' vehicles/day</div>' +

          '<div class="tip-box"><strong>Planner Tip:</strong> ' + tip + '</div>';
      }

      // HiDPI Forecast Canvas Chart
      var chartPointsMeta = [];
      function drawForecastChart() {
        var canvas = document.getElementById("forecast-canvas");
        if (!canvas || canvas.offsetParent === null) return;
        var rect = canvas.parentElement.getBoundingClientRect();
        var dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(300, Math.floor(rect.width * dpr));
        canvas.height = Math.max(220, Math.floor(rect.height * dpr));
        var ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);

        var W = rect.width;
        var H = rect.height;
        var pad = { top: 24, right: 26, bottom: 38, left: 56 };
        var plotW = W - pad.left - pad.right;
        var plotH = H - pad.top - pad.bottom;

        var mShort = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        function fmtShortDs(ds) {
          var pts = String(ds || "").split("-");
          if (pts.length !== 3) return ds;
          return (mShort[Number(pts[1]) - 1] || pts[1]) + " " + Number(pts[2]);
        }

        var histTail = state.showHist ? (PAYLOAD.recent_history || []).slice(-35) : [];
        var fcList = (PAYLOAD.forecast_days || []).slice().sort(function (a, b) { return a.ds < b.ds ? -1 : 1; });
        var series = [];
        histTail.forEach(function (h) {
          series.push({ ds: h.ds, label: fmtShortDs(h.ds), isHist: true, isWeekend: !!h.is_weekend, val: Number(h.vehicles), raw: h });
        });
        fcList.forEach(function (f) {
          series.push({
            ds: f.ds,
            label: f.date_short,
            isHist: false,
            isWeekend: !!f.is_weekend,
            val: Number(f.ensemble),
            pi80L: Number(f.pi80_low),
            pi80H: Number(f.pi80_high),
            pi95L: Number(f.pi95_low),
            pi95H: Number(f.pi95_high),
            raw: f
          });
        });

        if (!series.length) return;
        var yMin = Infinity, yMax = -Infinity;
        series.forEach(function (p) {
          var lo = (!p.isHist && state.showPi95) ? p.pi95L : (!p.isHist && state.showPi80) ? p.pi80L : p.val;
          var hi = (!p.isHist && state.showPi95) ? p.pi95H : (!p.isHist && state.showPi80) ? p.pi80H : p.val;
          if (lo < yMin) yMin = lo;
          if (hi > yMax) yMax = hi;
        });
        var ySpan = Math.max(1000, yMax - yMin);
        yMin = Math.max(0, Math.floor((yMin - ySpan * 0.12) / 1000) * 1000);
        yMax = Math.ceil((yMax + ySpan * 0.12) / 1000) * 1000;

        function xOf(i) { return pad.left + (series.length === 1 ? plotW / 2 : (i / (series.length - 1)) * plotW); }
        function yOf(v) { return pad.top + plotH - ((v - yMin) / Math.max(1, yMax - yMin)) * plotH; }

        ctx.clearRect(0, 0, W, H);
        var borderCol = cssVar("--border");
        var mutedCol = cssVar("--muted");
        var inkCol = cssVar("--ink");
        var glacierCol = cssVar("--glacier");
        var histCol = cssVar("--history");
        var bestCol = cssVar("--best");
        var badCol = cssVar("--bad");

        // Horizontal gridlines
        ctx.font = "600 11px " + cssVar("--font-sans");
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (var g = 0; g <= 4; g++) {
          var gv = yMin + (g / 4) * (yMax - yMin);
          var gy = yOf(gv);
          ctx.strokeStyle = borderCol;
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(W - pad.right, gy); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = mutedCol;
          ctx.fillText((gv / 1000).toFixed(1) + "k", pad.left - 8, gy);
        }

        // Weekend shading
        var stepW = series.length > 1 ? plotW / (series.length - 1) : plotW;
        if (state.showWknd) {
          ctx.fillStyle = cssVar("--weekend-band");
          series.forEach(function (p, i) {
            if (p.isWeekend) {
              var rx = Math.max(pad.left, xOf(i) - stepW * 0.48);
              var rw = Math.min(W - pad.right - rx, stepW * 0.96);
              ctx.fillRect(rx, pad.top, rw, plotH);
            }
          });
        }

        // Forecast transition vertical divider
        var firstFcIdx = -1;
        for (var k = 0; k < series.length; k++) {
          if (!series[k].isHist) { firstFcIdx = k; break; }
        }
        if (firstFcIdx > 0) {
          var divX = (xOf(firstFcIdx - 1) + xOf(firstFcIdx)) / 2;
          ctx.strokeStyle = cssVar("--border-strong");
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.beginPath(); ctx.moveTo(divX, pad.top); ctx.lineTo(divX, pad.top + plotH); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = mutedCol;
          ctx.textAlign = "left";
          ctx.fillText("Forecast →", divX + 6, pad.top + 8);
        }

        // 95% and 80% Conformal bands
        var fcIndices = [];
        series.forEach(function (p, i) { if (!p.isHist) fcIndices.push(i); });

        function drawBand(lowKey, highKey, fillStyle) {
          if (fcIndices.length < 2) return;
          ctx.fillStyle = fillStyle;
          ctx.beginPath();
          fcIndices.forEach(function (idx, pos) {
            var px = xOf(idx), py = yOf(series[idx][highKey]);
            if (pos === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          });
          for (var r = fcIndices.length - 1; r >= 0; r--) {
            var idx2 = fcIndices[r];
            ctx.lineTo(xOf(idx2), yOf(series[idx2][lowKey]));
          }
          ctx.closePath();
          ctx.fill();
        }
        if (state.showPi95) drawBand("pi95L", "pi95H", cssVar("--glacier-softer"));
        if (state.showPi80) drawBand("pi80L", "pi80H", cssVar("--glacier-soft"));

        // History line
        if (histTail.length > 1) {
          ctx.strokeStyle = histCol;
          ctx.lineWidth = 2;
          ctx.beginPath();
          for (var hi = 0; hi < firstFcIdx; hi++) {
            var hx = xOf(hi), hy = yOf(series[hi].val);
            if (hi === 0) ctx.moveTo(hx, hy); else ctx.lineTo(hx, hy);
          }
          ctx.stroke();
        }

        // Bridge dashed segment from last observed to first forecast
        if (firstFcIdx > 0) {
          ctx.strokeStyle = glacierCol;
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(xOf(firstFcIdx - 1), yOf(series[firstFcIdx - 1].val));
          ctx.lineTo(xOf(firstFcIdx), yOf(series[firstFcIdx].val));
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Forecast line
        if (fcIndices.length > 0) {
          ctx.strokeStyle = glacierCol;
          ctx.lineWidth = 3;
          ctx.beginPath();
          fcIndices.forEach(function (idx, pos) {
            var fx = xOf(idx), fy = yOf(series[idx].val);
            if (pos === 0) ctx.moveTo(fx, fy); else ctx.lineTo(fx, fy);
          });
          ctx.stroke();
        }

        // Highlight best, worst & selected
        var rec = L.computeRecommendations(PAYLOAD, state.dayFilter, state.priority, state.tripDays);
        var bestDs = rec && rec.bestDay ? rec.bestDay.ds : null;
        var worstDs = rec && rec.worstDay ? rec.worstDay.ds : null;

        chartPointsMeta = [];
        series.forEach(function (p, i) {
          var px = xOf(i), py = yOf(p.val);
          chartPointsMeta.push({ x: px, y: py, item: p });
          if (!p.isHist) {
            var isSel = p.ds === state.selectedDs;
            var isBest = p.ds === bestDs;
            var isWorst = p.ds === worstDs;
            if (isSel) {
              ctx.strokeStyle = cssVar("--accent");
              ctx.lineWidth = 1.5;
              ctx.setLineDash([2, 2]);
              ctx.beginPath(); ctx.moveTo(px, pad.top); ctx.lineTo(px, pad.top + plotH); ctx.stroke();
              ctx.setLineDash([]);
            }
            ctx.beginPath();
            ctx.arc(px, py, isSel ? 6.5 : (isBest || isWorst ? 5.5 : 3.8), 0, Math.PI * 2);
            ctx.fillStyle = isBest ? bestCol : isWorst ? badCol : glacierCol;
            ctx.fill();
            ctx.lineWidth = isSel ? 2.5 : 1.5;
            ctx.strokeStyle = cssVar("--surface");
            ctx.stroke();
          }
        });

        // X-axis date labels (evenly spaced without overlap)
        ctx.fillStyle = mutedCol;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        var maxTicks = Math.max(4, Math.floor(plotW / 76));
        var labelStep = Math.max(1, Math.ceil(series.length / maxTicks));
        var lastLblX = -Infinity;
        var endLblX = xOf(series.length - 1);
        series.forEach(function (p, i) {
          var isEnd = (i === series.length - 1);
          if (i % labelStep === 0 || isEnd) {
            var px = xOf(i);
            if (!isEnd && (px - lastLblX < 54 || endLblX - px < 54)) return;
            ctx.fillText(p.label, px, pad.top + plotH + 10);
            lastLblX = px;
          }
        });
      }

      // Canvas Hover & Click
      var fCanvas = document.getElementById("forecast-canvas");
      var fTip = document.getElementById("chart-tooltip");
      fCanvas.addEventListener("mousemove", function (ev) {
        if (!chartPointsMeta.length) return;
        var r = fCanvas.getBoundingClientRect();
        var mx = ev.clientX - r.left;
        var nearest = null, bestDist = Infinity;
        chartPointsMeta.forEach(function (pt) {
          var d = Math.abs(pt.x - mx);
          if (d < bestDist) { bestDist = d; nearest = pt; }
        });
        if (nearest && bestDist < 32) {
          var p = nearest.item;
          var txt = '<strong>' + p.ds + (p.isHist ? " (Observed)" : " (Forecast)") + '</strong><br/>' +
            '<span class="tabular">' + fmtNum(p.val) + ' vehicles</span>';
          if (!p.isHist && p.raw) {
            txt += '<br/><span class="tabular" style="color:var(--muted);">80% PI: ' + fmtNum(p.pi80L) + '–' + fmtNum(p.pi80H) + '</span>' +
                   '<br/>' + (p.raw.weather_label || "") + ' · ' + fmtTemp(p.raw.temp_max_c);
            if (p.raw.event_label) txt += '<br/><span style="color:var(--accent);">' + p.raw.event_label + '</span>';
          }
          fTip.innerHTML = txt;
          var leftPos = Math.min(r.width - 210, Math.max(12, nearest.x + 12));
          var topPos = Math.max(8, nearest.y - 54);
          fTip.style.left = leftPos + "px";
          fTip.style.top = topPos + "px";
          fTip.style.opacity = "1";
        } else {
          fTip.style.opacity = "0";
        }
      });
      fCanvas.addEventListener("mouseleave", function () { fTip.style.opacity = "0"; });
      fCanvas.addEventListener("click", function (ev) {
        if (!chartPointsMeta.length) return;
        var r = fCanvas.getBoundingClientRect();
        var mx = ev.clientX - r.left;
        var nearest = null, bestDist = Infinity;
        chartPointsMeta.forEach(function (pt) {
          if (pt.item.isHist || pt.item.raw.is_bridge_gap) return;
          var d = Math.abs(pt.x - mx);
          if (d < bestDist) { bestDist = d; nearest = pt; }
        });
        if (nearest && bestDist < 28) {
          state.selectedDs = nearest.item.ds;
          renderStrip();
          renderInspector();
          drawForecastChart();
        }
      });

      // View 2: Annual 365-Day Calendar Heatmap & Dow/Monthly Charts
      function buildAnnualSection() {
        var annual = PAYLOAD.annual_planner || {};
        var prof = annual.daily_profile_365 || [];
        var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        var byMd = {};
        var vMin = Infinity, vMax = -Infinity;
        prof.forEach(function (r) {
          byMd[r.month + "-" + r.day] = r;
          if (r.avg_vehicles < vMin) vMin = r.avg_vehicles;
          if (r.avg_vehicles > vMax) vMax = r.avg_vehicles;
        });

        function heatColor(v) {
          var t = Math.max(0, Math.min(1, (v - vMin) / Math.max(1, vMax - vMin)));
          if (t < 0.35) {
            return "hsl(" + Math.round(145 - t * 40) + ", 55%, " + Math.round(78 - t * 35) + "%)";
          }
          if (t < 0.7) {
            return "hsl(" + Math.round(52 - (t - 0.35) * 60) + ", 80%, " + Math.round(62 - (t - 0.35) * 20) + "%)";
          }
          return "hsl(" + Math.round(24 - (t - 0.7) * 55) + ", 78%, " + Math.round(54 - (t - 0.7) * 24) + "%)";
        }

        var rowsHtml = "";
        for (var m = 1; m <= 12; m++) {
          rowsHtml += '<div class="hm-row"><span class="hm-month">' + months[m - 1] + '</span>';
          for (var d = 1; d <= 31; d++) {
            var item = byMd[m + "-" + d];
            if (!item) {
              rowsHtml += '<span class="hm-cell hm-empty"></span>';
            } else {
              rowsHtml += '<button type="button" class="hm-cell" style="background:' + heatColor(item.avg_vehicles) +
                ';" data-md="' + m + '-' + d + '" aria-label="' + item.date_label + ' average ' + fmtNum(item.avg_vehicles) + ' vehicles"></button>';
            }
          }
          rowsHtml += '</div>';
        }
        var hmEl = document.getElementById("annual-heatmap");
        hmEl.innerHTML = rowsHtml;
        hmEl.querySelectorAll(".hm-cell[data-md]").forEach(function (cell) {
          var show = function () {
            var item = byMd[cell.getAttribute("data-md")];
            if (!item) return;
            document.getElementById("hm-readout").innerHTML =
              '<strong>' + item.date_label + ' (' + item.season_tag + ')</strong>' +
              '<span class="tabular">3-Yr Avg: <strong>' + fmtNum(item.avg_vehicles) + ' veh/day</strong> (IQR ' +
              fmtNum(item.p25_vehicles) + '–' + fmtNum(item.p75_vehicles) + ') · Normal High/Low: ' +
              fmtTemp(item.temp_max_c) + ' / ' + fmtTemp(item.temp_min_c) + '</span>';
          };
          cell.addEventListener("mouseenter", show);
          cell.addEventListener("click", show);
        });

        var ins = PAYLOAD.insights || {};
        document.getElementById("insight-weekly-text").textContent = ins.weekly_insight || "";
        document.getElementById("insight-seasonal-text").textContent = ins.seasonal_insight || "";
      }

      function drawBarChart(canvasId, labels, values, colors, suffix) {
        var canvas = document.getElementById(canvasId);
        if (!canvas || canvas.offsetParent === null || !values.length) return;
        var rect = canvas.parentElement.getBoundingClientRect();
        var dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(260, Math.floor(rect.width * dpr));
        canvas.height = Math.max(200, Math.floor(rect.height * dpr));
        var ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);
        var W = rect.width, H = rect.height;
        var pad = { top: 24, right: 14, bottom: 32, left: 48 };
        var pW = W - pad.left - pad.right, pH = H - pad.top - pad.bottom;
        var maxV = Math.max.apply(null, values) * 1.12;
        var borderCol = cssVar("--border"), mutedCol = cssVar("--muted"), inkCol = cssVar("--ink");

        ctx.font = "600 11px " + cssVar("--font-sans");
        for (var g = 0; g <= 3; g++) {
          var v = (g / 3) * maxV;
          var y = pad.top + pH - (v / maxV) * pH;
          ctx.strokeStyle = borderCol; ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = mutedCol; ctx.textAlign = "right"; ctx.textBaseline = "middle";
          ctx.fillText((v / 1000).toFixed(0) + "k", pad.left - 6, y);
        }

        var slot = pW / labels.length;
        var bw = Math.min(34, slot * 0.62);
        labels.forEach(function (lbl, i) {
          var bh = (values[i] / maxV) * pH;
          var bx = pad.left + i * slot + (slot - bw) / 2;
          var by = pad.top + pH - bh;
          ctx.fillStyle = colors[i];
          ctx.beginPath();
          ctx.roundRect ? ctx.roundRect(bx, by, bw, bh, [5, 5, 0, 0]) : ctx.rect(bx, by, bw, bh);
          ctx.fill();
          ctx.fillStyle = inkCol; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
          ctx.fillText((values[i] / 1000).toFixed(1) + suffix, bx + bw / 2, by - 4);
          ctx.fillStyle = mutedCol; ctx.textBaseline = "top";
          ctx.fillText(lbl, bx + bw / 2, pad.top + pH + 8);
        });
      }

      function drawAnnualCharts() {
        var annual = PAYLOAD.annual_planner || {};
        var dow = annual.dow_profile || [];
        var mon = annual.monthly_typical || [];
        drawBarChart(
          "dow-canvas",
          dow.map(function (d) { return d.short; }),
          dow.map(function (d) { return d.avg_vehicles; }),
          dow.map(function (d) { return d.is_weekend ? cssVar("--warn") : cssVar("--accent"); }),
          "k"
        );
        drawBarChart(
          "monthly-canvas",
          mon.map(function (m) { return m.name; }),
          mon.map(function (m) { return m.avg_vehicles; }),
          mon.map(function (m) { return m.avg_vehicles > 22000 ? cssVar("--warn") : cssVar("--glacier"); }),
          "k"
        );
      }

      // View 3: Science & Live Sources
      function buildScienceSection() {
        var cv = PAYLOAD.cv_benchmark || {};
        document.getElementById("cv-subtitle").textContent =
          (cv.n_folds || 12) + " expanding-window folds × " + (cv.horizon_days || 16) + "-day horizons (" + (cv.n_oof_days || 192) + " unseen days across all 4 Banff seasons)";
        document.getElementById("cv-cov-badge").textContent =
          "Conformal Cov: 80% → " + (cv.empirical_cov80_pct || 80) + "% · 95% → " + (cv.empirical_cov95_pct || 95) + "%";

        var rows = (cv.summary_metrics || []).map(function (m) {
          var isEns = m.model === "Ensemble_Weighted";
          return '<tr class="' + (isEns ? "is-ensemble" : "") + '">' +
            '<td>' + m.model + '</td>' +
            '<td class="tabular">' + fmtNum(m.MAE) + '</td>' +
            '<td class="tabular">' + fmtNum(m.RMSE) + '</td>' +
            '<td class="tabular">' + Number(m.MAPE).toFixed(2) + '%</td>' +
            '<td class="tabular">' + Number(m.sMAPE).toFixed(2) + '%</td>' +
            '<td class="tabular">' + Number(m.MASE).toFixed(3) + '</td>' +
            '<td class="tabular">' + (m.Bias_pct >= 0 ? "+" : "") + Number(m.Bias_pct).toFixed(2) + '%</td>' +
          '</tr>';
        }).join("");
        document.getElementById("cv-table").innerHTML =
          '<thead><tr><th>Model Family</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>sMAPE</th><th>MASE</th><th>Bias</th></tr></thead><tbody>' + rows + '</tbody>';

        var weights = cv.ensemble_weights || {};
        var wHtml = Object.keys(weights).sort(function (a, b) { return weights[b] - weights[a]; }).map(function (m) {
          var pct = Math.round(weights[m] * 1000) / 10;
          return '<div class="pct-bar-wrap">' +
            '<div class="pct-bar-labels"><span>' + m + '</span><strong class="tabular">' + pct.toFixed(1) + '%</strong></div>' +
            '<div class="pct-track"><div class="pct-fill" style="width:' + Math.max(2, pct) + '%;background:var(--accent);"></div></div>' +
          '</div>';
        }).join("");
        document.getElementById("weights-bars").innerHTML = wHtml;

        var f = PAYLOAD.data_freshness || {};
        var mac = f.latest_macro_snapshot || {};
        var cards = [
          {
            title: "Town of Banff Traffic Counters",
            meta: "Tableau Public (.twb → .hyper) · " + fmtNum(f.traffic_days) + " complete days (" + f.traffic_start_date + " to " + f.traffic_last_observed + "). Tracks West, East &amp; Mountain Ave (>=23h/day &amp; 4x guards)."
          },
          {
            title: "Banff Weather &amp; 16-Day NWP",
            meta: "ECCC BANFF CS (3050519) + Open-Meteo ERA5 bias-calibrated archive + 16-day forecast through " + f.weather_forecast_through + "."
          },
          {
            title: "Bank of Canada FX (USD/CAD)",
            meta: "Daily Valet FXUSDCAD (legacy pre-2017 IEXE0101 spliced) observed through " + f.fx_last_observed + (f.fx_latest_usdcad ? " (latest: " + f.fx_latest_usdcad + ")" : "") + "."
          },
          {
            title: "Statistics Canada Macro &amp; Visitors",
            meta: "Monthly CPI Gasoline, Unemployment (" + (mac.AB_LFS_UnempRate_SA || "—") + "%), Food Services &amp; Int'l Travellers through " + f.statcan_macro_through + "."
          },
          {
            title: "Alberta Economic Dashboard",
            meta: "Hwy 1 Banff Park Gates (" + f.alberta_hwy1_through + ") &amp; Banff National Park Attendance (" + f.alberta_park_attendance_through + ")."
          },
          {
            title: "Scope Governance",
            meta: "Alberta statutory &amp; bridge calendar integrated natively. Municipal Zero Waste series are strictly excluded from all models &amp; features."
          }
        ];
        document.getElementById("sources-grid").innerHTML = cards.map(function (c) {
          return '<div class="source-card"><h4 class="source-title">' + c.title + '</h4><p class="source-meta">' + c.meta + '</p></div>';
        }).join("");
      }

      function drawChartsForActiveView() {
        if (state.view === "planner") drawForecastChart();
        if (state.view === "annual") drawAnnualCharts();
      }

      buildAnnualSection();
      buildScienceSection();
      renderPlanner();
      if (qView || hashView) setActiveView(qView || hashView);

      window.addEventListener("resize", drawChartsForActiveView);
    })();
  </script>
</body>
</html>
"""


def build_static_site(payload: Dict, site_dir: Path = SITE_DIR) -> Path:
    """Render self-contained static HTML application to site/index.html + site/data/forecast_payload.json."""
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir = site_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Escape </script> inside inline JSON cleanly for strict HTML5 safety
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (data_dir / "forecast_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    html = _HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)
    out_html = site_dir / "index.html"
    out_html.write_text(html, encoding="utf-8")
    # Also write .nojekyll so GitHub Pages skips Jekyll processing cleanly
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    return out_html
