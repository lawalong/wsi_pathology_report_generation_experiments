"""
Merge all experiment results into a single paper-ready table.

Reads metrics_test.json, metrics_val.json, and semantic_errors_test.json
from each experiment under runs/, and outputs:
  - results/main_table.csv        (comma-separated)
  - results/main_table_latex.tex  (LaTeX booktabs table)
  - results/main_table.json       (machine-readable)
  - printed ASCII table

Usage:
  python3 merge_results_for_paper.py
  python3 merge_results_for_paper.py --split test
  python3 merge_results_for_paper.py --split val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
try:
    from config import OUTPUT_DIR
    RUNS_DIR = OUTPUT_DIR.parent / "runs"
except ImportError:
    RUNS_DIR = Path(__file__).parent / "runs"

RESULTS_DIR = Path(__file__).parent / "results"

# Experiments to include, in display order
# (folder_name, display_name, type)
# type: "text" = has ROUGE/KW metrics, "classifier" = has macro_f1/micro_f1
EXPERIMENTS: List[tuple] = [
    ("baseline_retrieval",      "Raw Retrieval",           "text"),
    ("baseline_structured",     "Structured (top-1)",      "text"),
    ("topk_fusion_k3",          "Fusion k=3",              "text"),
    ("topk_fusion_k5",          "Fusion k=5",              "text"),
    ("rerank_k1",               "Rerank k=1",              "text"),
    ("rerank_k3",               "Rerank+Fusion k=3",       "text"),
    ("ours_prompt_semantic",    "Ours (T5+Vis+Sem)",       "text"),
    ("ours_prompt_semantic_v1", "Ours v1 (no prompt fix)", "text"),
    ("ours_prompt_semantic_v3", "Ours v3 (gen-text sem)",  "text"),
    ("modular_classifier",      "Modular Classifier",      "classifier"),
]


# ============================================================================
# Helpers
# ============================================================================

def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v, decimals: int = 4) -> str:
    """Format a number or return '—' for None."""
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def fmt_pct(v, decimals: int = 1) -> str:
    """Format as percentage or return '—'."""
    if v is None:
        return "—"
    return f"{v*100:.{decimals}f}%"


# ============================================================================
# Build rows
# ============================================================================

def build_rows(split: str) -> List[Dict]:
    """Build one row per experiment with all available metrics."""
    rows = []
    for folder, display_name, exp_type in EXPERIMENTS:
        run_dir = RUNS_DIR / folder

        # -- Text-generation metrics (ROUGE / KW) --
        metrics = load_json(run_dir / f"metrics_{split}.json")

        # -- Semantic error metrics --
        sem = load_json(run_dir / f"semantic_errors_{split}.json")

        # -- Classifier metrics --
        row: Dict = {
            "experiment": folder,
            "method": display_name,
            "type": exp_type,
        }

        if metrics:
            if exp_type == "text":
                row["rouge1"] = metrics.get("rouge1")
                row["rouge2"] = metrics.get("rouge2")
                row["rougeL"] = metrics.get("rougeL")
                row["kw_cov"] = metrics.get("keyword_coverage")
            elif exp_type == "classifier":
                row["macro_f1"] = metrics.get("macro_f1")
                row["micro_f1"] = metrics.get("micro_f1")

        if sem and "global" in sem:
            g = sem["global"]
            row["sem_match"] = g.get("match_rate")
            row["sem_missed"] = g.get("missed_rate")
            row["sem_wrong"] = g.get("wrong_rate")

        rows.append(row)

    return rows


# ============================================================================
# Output formatters
# ============================================================================

TEXT_COLUMNS = [
    ("method",     "Method",       20),
    ("rouge1",     "ROUGE-1",       8),
    ("rouge2",     "ROUGE-2",       8),
    ("rougeL",     "ROUGE-L",       8),
    ("kw_cov",     "KW Cov",        8),
    ("sem_match",  "Sem Match↑",   10),
    ("sem_missed", "Sem Miss↓",    10),
    ("sem_wrong",  "Sem Wrong↓",   10),
]

CLF_COLUMNS = [
    ("method",    "Method",      20),
    ("macro_f1",  "Macro F1",     9),
    ("micro_f1",  "Micro F1",     9),
]


def print_ascii(rows: List[Dict], split: str) -> None:
    """Print a nicely formatted ASCII table."""
    text_rows = [r for r in rows if r["type"] == "text" and any(
        r.get(c[0]) is not None for c in TEXT_COLUMNS[1:])]
    clf_rows = [r for r in rows if r["type"] == "classifier"]

    if text_rows:
        print(f"\n{'='*90}")
        print(f"  Main Results — {split.upper()} set (Text Generation)")
        print(f"{'='*90}")

        # Header
        hdr = "  "
        for key, label, width in TEXT_COLUMNS:
            hdr += f"{label:>{width}s}  "
        print(hdr)
        print("  " + "-" * (sum(w + 2 for _, _, w in TEXT_COLUMNS)))

        for r in text_rows:
            line = "  "
            for key, _, width in TEXT_COLUMNS:
                v = r.get(key)
                if key == "method":
                    line += f"{v:<{width}s}  "
                elif v is not None:
                    line += f"{v:>{width}.4f}  "
                else:
                    line += f"{'—':>{width}s}  "
            print(line)

    if clf_rows:
        print(f"\n{'='*50}")
        print(f"  Classifier Results — {split.upper()} set")
        print(f"{'='*50}")
        hdr = "  "
        for key, label, width in CLF_COLUMNS:
            hdr += f"{label:>{width}s}  "
        print(hdr)
        print("  " + "-" * (sum(w + 2 for _, _, w in CLF_COLUMNS)))
        for r in clf_rows:
            line = "  "
            for key, _, width in CLF_COLUMNS:
                v = r.get(key)
                if key == "method":
                    line += f"{v:<{width}s}  "
                elif v is not None:
                    line += f"{v:>{width}.4f}  "
                else:
                    line += f"{'—':>{width}s}  "
            print(line)

    print()


def save_csv(rows: List[Dict], path: Path) -> None:
    """Save results as CSV."""
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)


def save_latex(rows: List[Dict], path: Path) -> None:
    """Save a LaTeX booktabs table for text-generation experiments."""
    text_rows = [r for r in rows if r["type"] == "text" and any(
        r.get(c) is not None for c in ["rouge1", "rougeL", "kw_cov"])]

    if not text_rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main results on TCGA-BRCA test set.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccc|ccc}",
        r"\toprule",
        r"Method & R-1 & R-2 & R-L & KW\% & Match$\uparrow$ & Miss$\downarrow$ & Wrong$\downarrow$ \\",
        r"\midrule",
    ]

    for r in text_rows:
        cols = [
            r["method"],
            fmt(r.get("rouge1")),
            fmt(r.get("rouge2")),
            fmt(r.get("rougeL")),
            fmt(r.get("kw_cov")),
            fmt_pct(r.get("sem_match")),
            fmt_pct(r.get("sem_missed")),
            fmt_pct(r.get("sem_wrong")),
        ]
        lines.append(" & ".join(cols) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge experiment results into paper-ready tables.",
    )
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val"])
    args = parser.parse_args()

    rows = build_rows(args.split)

    # Print
    print_ascii(rows, args.split)

    # Save
    csv_path = RESULTS_DIR / f"main_table_{args.split}.csv"
    save_csv(rows, csv_path)
    print(f"  ✅ CSV:   {csv_path}")

    json_path = RESULTS_DIR / f"main_table_{args.split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"  ✅ JSON:  {json_path}")

    tex_path = RESULTS_DIR / f"main_table_{args.split}.tex"
    save_latex(rows, tex_path)
    if tex_path.exists():
        print(f"  ✅ LaTeX: {tex_path}")

    print()


if __name__ == "__main__":
    main()
