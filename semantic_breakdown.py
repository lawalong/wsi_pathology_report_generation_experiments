"""
Semantic Breakdown — per-case and per-concept error analysis.

For each experiment, produces a detailed breakdown of:
  - wrong:   concepts predicted but NOT in GT (hallucinations)
  - missing: concepts in GT but NOT in prediction
  - extra:   same as wrong (alias for paper clarity)

Reuses the same concept set and matching logic as compute_semantic_errors.py.

Outputs:
  runs/{exp}/semantic_breakdown_{split}.json   — per-case details
  Printed summary table

Usage:
  python3 semantic_breakdown.py --exp ours_prompt_semantic
  python3 semantic_breakdown.py --exp baseline_structured
  python3 semantic_breakdown.py   # runs both and prints side-by-side
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Reuse concept set from compute_semantic_errors.py
# ---------------------------------------------------------------------------
CONCEPTS = {
    "carcinoma":  [r"\bcarcinoma\b"],
    "benign":     [r"\bbenign\b"],
    "negative":   [r"\bnegative\b"],
    "positive":   [r"\bpositive\b"],
    "metastasis": [r"\bmetasta\w*\b"],
    "invasive":   [r"\binvasive\b", r"\binfiltrating\b"],
    "ductal":     [r"\bductal\b"],
    "lobular":    [r"\blobular\b"],
    "grade":      [r"\bgrade\s*[123iIvV]", r"\bnottingham\b"],
    "in_situ":    [r"\bin\s*situ\b", r"\bdcis\b"],
    "lymph_node": [r"\blymph\s*node"],
    "margin":     [r"\bmargin"],
}

RUNS_DIR = Path(__file__).parent / "runs"


def concept_present(text: str, patterns: list) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def analyse_case(case_id: str, gt: str, pred: str) -> Dict:
    """Return per-concept match/missed/wrong + aggregated counts."""
    concepts_detail = {}
    n_match = 0
    n_missed = 0
    n_wrong = 0

    wrong_list = []
    missing_list = []
    match_list = []

    for concept, patterns in CONCEPTS.items():
        in_gt = concept_present(gt, patterns)
        in_pred = concept_present(pred, patterns)

        if in_gt and in_pred:
            status = "match"
            n_match += 1
            match_list.append(concept)
        elif in_gt and not in_pred:
            status = "missing"
            n_missed += 1
            missing_list.append(concept)
        elif not in_gt and in_pred:
            status = "wrong"
            n_wrong += 1
            wrong_list.append(concept)
        else:
            status = "absent"

        concepts_detail[concept] = status

    return {
        "case_id": case_id,
        "n_match": n_match,
        "n_missing": n_missed,
        "n_wrong": n_wrong,
        "match_concepts": match_list,
        "missing_concepts": missing_list,
        "wrong_concepts": wrong_list,
        "detail": concepts_detail,
    }


def run_breakdown(exp_name: str, split: str = "test") -> Optional[Dict]:
    """Run full breakdown on one experiment."""
    pred_path = RUNS_DIR / exp_name / f"predictions_{split}.jsonl"
    if not pred_path.exists():
        print(f"  ❌ Not found: {pred_path}")
        return None

    with open(pred_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    cases = []
    concept_agg = defaultdict(lambda: {"match": 0, "missing": 0, "wrong": 0, "absent": 0})

    for s in samples:
        case = analyse_case(s["case_id"], s["gt"], s["pred"])
        cases.append(case)
        for concept, status in case["detail"].items():
            concept_agg[concept][status] += 1

    # Global counts
    total_wrong = sum(c["n_wrong"] for c in cases)
    total_missing = sum(c["n_missing"] for c in cases)
    total_match = sum(c["n_match"] for c in cases)
    total_relevant = total_match + total_missing

    result = {
        "experiment": exp_name,
        "split": split,
        "n_cases": len(samples),
        "summary": {
            "total_match": total_match,
            "total_missing": total_missing,
            "total_wrong": total_wrong,
            "match_rate": round(total_match / total_relevant, 4) if total_relevant else 0,
            "missing_rate": round(total_missing / total_relevant, 4) if total_relevant else 0,
            "wrong_rate": round(total_wrong / (len(samples) * len(CONCEPTS)), 4) if samples else 0,
        },
        "per_concept": {
            concept: dict(concept_agg[concept]) for concept in CONCEPTS
        },
        "per_case": cases,
    }
    return result


def print_breakdown(res: Dict) -> None:
    """Print readable summary."""
    s = res["summary"]
    print(f"\n{'='*65}")
    print(f"  {res['experiment']} — {res['split']} ({res['n_cases']} cases)")
    print(f"{'='*65}")
    print(f"  Match: {s['total_match']}  Missing: {s['total_missing']}  "
          f"Wrong: {s['total_wrong']}")
    print(f"  Rates: match={s['match_rate']:.1%}  "
          f"missing={s['missing_rate']:.1%}  "
          f"wrong={s['wrong_rate']:.1%}")

    print(f"\n  {'Concept':<12} {'Match':>6} {'Missing':>8} {'Wrong':>6} {'Absent':>7}")
    print(f"  {'-'*42}")
    for concept in CONCEPTS:
        c = res["per_concept"][concept]
        print(f"  {concept:<12} {c['match']:>6} {c['missing']:>8} {c['wrong']:>6} {c['absent']:>7}")

    # Worst cases by wrong count
    worst_wrong = sorted(res["per_case"], key=lambda x: x["n_wrong"], reverse=True)
    print(f"\n  Top 5 cases by WRONG concepts (hallucinations):")
    for case in worst_wrong[:5]:
        if case["n_wrong"] == 0:
            break
        print(f"    {case['case_id']}: wrong={case['wrong_concepts']}")

    # Worst cases by missing count
    worst_missing = sorted(res["per_case"], key=lambda x: x["n_missing"], reverse=True)
    print(f"\n  Top 5 cases by MISSING concepts:")
    for case in worst_missing[:5]:
        if case["n_missing"] == 0:
            break
        print(f"    {case['case_id']}: missing={case['missing_concepts']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic breakdown: per-case wrong/missing/match analysis.",
    )
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment name. If omitted, runs baseline_structured "
                             "and ours_prompt_semantic.")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    if args.exp:
        exps = [args.exp]
    else:
        exps = ["baseline_structured", "ours_prompt_semantic"]

    for exp_name in exps:
        res = run_breakdown(exp_name, args.split)
        if res is None:
            continue

        print_breakdown(res)

        # Save
        out_path = RUNS_DIR / exp_name / f"semantic_breakdown_{args.split}.json"
        save_data = {k: v for k, v in res.items() if k != "per_case"}
        save_data["per_case_summary"] = [
            {k: v for k, v in c.items() if k != "detail"}
            for c in res["per_case"]
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"\n  ✅ Saved: {out_path}")

    print()


if __name__ == "__main__":
    main()
