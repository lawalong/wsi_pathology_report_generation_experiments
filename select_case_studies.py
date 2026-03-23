"""
Select Case Studies — find cases with the largest semantic improvement.

Compares two experiments (default: baseline_structured vs ours_prompt_semantic)
and identifies cases where the proposed method reduces semantic errors the most.

Ranks by:
  1. Reduction in "wrong" concepts (hallucination reduction)
  2. Reduction in "missing" concepts (recall improvement)
  3. Net improvement = (wrong_reduction + missing_reduction)

Outputs:
  results/case_studies.json
  Printed top-N cases with GT/pred excerpts

Usage:
  python3 select_case_studies.py
  python3 select_case_studies.py --baseline baseline_structured --ours ours_prompt_semantic --top 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Reuse concept set
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
RESULTS_DIR = Path(__file__).parent / "results"


def concept_present(text: str, patterns: list) -> bool:
    return any(re.search(p, text.lower()) for p in patterns)


def get_concept_status(gt: str, pred: str) -> Dict[str, str]:
    """Return {concept: match/missed/wrong/absent} for one case."""
    result = {}
    for concept, patterns in CONCEPTS.items():
        in_gt = concept_present(gt, patterns)
        in_pred = concept_present(pred, patterns)
        if in_gt and in_pred:
            result[concept] = "match"
        elif in_gt and not in_pred:
            result[concept] = "missed"
        elif not in_gt and in_pred:
            result[concept] = "wrong"
        else:
            result[concept] = "absent"
    return result


def load_predictions(exp_name: str, split: str) -> Dict[str, Dict]:
    """Load predictions keyed by case_id."""
    path = RUNS_DIR / exp_name / f"predictions_{split}.jsonl"
    if not path.exists():
        print(f"  ❌ Not found: {path}")
        return {}
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["case_id"]] = rec
    return preds


def compare_cases(
    baseline_preds: Dict[str, Dict],
    ours_preds: Dict[str, Dict],
) -> List[Dict]:
    """Compare semantic errors between baseline and ours for shared cases."""
    # Find shared case_ids
    shared = sorted(set(baseline_preds.keys()) & set(ours_preds.keys()))
    if not shared:
        print("  ⚠ No shared case_ids between experiments.")
        return []

    results = []
    for cid in shared:
        b = baseline_preds[cid]
        o = ours_preds[cid]

        gt = b["gt"]  # GT should be the same
        b_status = get_concept_status(gt, b["pred"])
        o_status = get_concept_status(gt, o["pred"])

        b_wrong = [c for c, s in b_status.items() if s == "wrong"]
        o_wrong = [c for c, s in o_status.items() if s == "wrong"]
        b_missed = [c for c, s in b_status.items() if s == "missed"]
        o_missed = [c for c, s in o_status.items() if s == "missed"]

        wrong_reduction = len(b_wrong) - len(o_wrong)
        missing_reduction = len(b_missed) - len(o_missed)
        net_improvement = wrong_reduction + missing_reduction

        # Concepts that flipped from wrong→correct or wrong→absent
        fixed_wrong = [c for c in b_wrong if c not in o_wrong]
        # Concepts that were missed in baseline but matched in ours
        fixed_missed = [c for c in b_missed if o_status.get(c) == "match"]
        # Concepts that regressed (new wrongs or new misses)
        new_wrong = [c for c in o_wrong if c not in b_wrong]
        new_missed = [c for c in o_missed if c not in b_missed]

        results.append({
            "case_id": cid,
            "wrong_reduction": wrong_reduction,
            "missing_reduction": missing_reduction,
            "net_improvement": net_improvement,
            "baseline_wrong": b_wrong,
            "ours_wrong": o_wrong,
            "baseline_missed": b_missed,
            "ours_missed": o_missed,
            "fixed_wrong": fixed_wrong,
            "fixed_missed": fixed_missed,
            "new_wrong": new_wrong,
            "new_missed": new_missed,
            "gt_excerpt": gt[:200],
            "baseline_pred_excerpt": b["pred"][:200],
            "ours_pred_excerpt": o["pred"][:200],
        })

    return results


def print_case_studies(cases: List[Dict], top_n: int,
                       baseline_name: str, ours_name: str) -> None:
    """Print the top case studies."""

    # Sort by net improvement (hallucination reduction prioritised)
    cases_sorted = sorted(cases, key=lambda x: (
        x["wrong_reduction"], x["net_improvement"]
    ), reverse=True)

    print(f"\n{'='*70}")
    print(f"  TOP {top_n} CASE STUDIES: Semantic Improvement")
    print(f"  {baseline_name}  →  {ours_name}")
    print(f"{'='*70}")

    for i, case in enumerate(cases_sorted[:top_n], 1):
        cid = case["case_id"]
        print(f"\n  {'─'*60}")
        print(f"  #{i}  {cid}")
        print(f"  Wrong reduction:   {case['wrong_reduction']:+d}  "
              f"({len(case['baseline_wrong'])} → {len(case['ours_wrong'])})")
        print(f"  Missing reduction: {case['missing_reduction']:+d}  "
              f"({len(case['baseline_missed'])} → {len(case['ours_missed'])})")
        print(f"  Net improvement:   {case['net_improvement']:+d}")

        if case["fixed_wrong"]:
            print(f"  ✅ Fixed wrong:   {case['fixed_wrong']}")
        if case["fixed_missed"]:
            print(f"  ✅ Fixed missed:  {case['fixed_missed']}")
        if case["new_wrong"]:
            print(f"  ❌ New wrong:     {case['new_wrong']}")
        if case["new_missed"]:
            print(f"  ❌ New missed:    {case['new_missed']}")

        print(f"\n  GT:       {case['gt_excerpt']}...")
        print(f"  Baseline: {case['baseline_pred_excerpt']}...")
        print(f"  Ours:     {case['ours_pred_excerpt']}...")

    # Also show worst regressions
    cases_worst = sorted(cases, key=lambda x: x["net_improvement"])
    regressions = [c for c in cases_worst if c["net_improvement"] < 0]
    if regressions:
        print(f"\n\n{'='*70}")
        print(f"  ⚠ REGRESSIONS (ours worse than baseline): {len(regressions)} cases")
        print(f"{'='*70}")
        for case in regressions[:3]:
            print(f"  {case['case_id']}: net={case['net_improvement']:+d}  "
                  f"new_wrong={case['new_wrong']}  new_missed={case['new_missed']}")

    # Summary stats
    improvements = sum(1 for c in cases if c["net_improvement"] > 0)
    ties = sum(1 for c in cases if c["net_improvement"] == 0)
    worse = sum(1 for c in cases if c["net_improvement"] < 0)
    print(f"\n  Summary: {improvements} improved, {ties} tied, {worse} regressed "
          f"(out of {len(cases)} shared cases)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select best case studies showing semantic improvement.",
    )
    parser.add_argument("--baseline", type=str, default="baseline_structured")
    parser.add_argument("--ours", type=str, default="ours_prompt_semantic")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading {args.baseline}...")
    baseline_preds = load_predictions(args.baseline, args.split)
    print(f"Loading {args.ours}...")
    ours_preds = load_predictions(args.ours, args.split)

    if not baseline_preds or not ours_preds:
        return

    cases = compare_cases(baseline_preds, ours_preds)
    print_case_studies(cases, args.top, args.baseline, args.ours)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "case_studies.json"
    # Save without long excerpts for cleaner JSON
    save_data = []
    for c in sorted(cases, key=lambda x: x["net_improvement"], reverse=True):
        save_data.append({
            "case_id": c["case_id"],
            "wrong_reduction": c["wrong_reduction"],
            "missing_reduction": c["missing_reduction"],
            "net_improvement": c["net_improvement"],
            "fixed_wrong": c["fixed_wrong"],
            "fixed_missed": c["fixed_missed"],
            "new_wrong": c["new_wrong"],
            "new_missed": c["new_missed"],
            "baseline_wrong": c["baseline_wrong"],
            "ours_wrong": c["ours_wrong"],
            "baseline_missed": c["baseline_missed"],
            "ours_missed": c["ours_missed"],
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Saved: {out_path}")
    print()


if __name__ == "__main__":
    main()
