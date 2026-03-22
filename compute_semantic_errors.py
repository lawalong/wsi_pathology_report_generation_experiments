"""
Semantic Error Quantification — Phase 4

For each test case, check whether clinically important concepts
appear in the GT and/or in the predicted text.

Metrics (per-concept, per-case):
  - Match:  concept in GT AND in pred   (correct)
  - Missed: concept in GT but NOT pred  (false negative)
  - Wrong:  concept NOT in GT but in pred (false positive / hallucination)

Runs on any experiment that has predictions_test.jsonl.

Usage:
  python3 compute_semantic_errors.py --exp baseline_structured
  python3 compute_semantic_errors.py --exp ours_prompt_semantic
  python3 compute_semantic_errors.py          # runs both and compares
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# ── Concept set (locked) ────────────────────────────────────────
CONCEPTS = {
    "carcinoma":  [r"\bcarcinoma\b"],
    "benign":     [r"\bbenign\b"],
    "negative":   [r"\bnegative\b"],
    "positive":   [r"\bpositive\b"],
    "metastasis": [r"\bmetasta\w*\b"],          # metastasis, metastatic
    "invasive":   [r"\binvasive\b", r"\binfiltrating\b"],
    "ductal":     [r"\bductal\b"],
    "lobular":    [r"\blobular\b"],
    "grade":      [r"\bgrade\s*[123iIvV]", r"\bnottingham\b"],
    "in_situ":    [r"\bin\s*situ\b", r"\bdcis\b"],
    "lymph_node": [r"\blymph\s*node"],
    "margin":     [r"\bmargin"],
}


def concept_present(text: str, patterns: list[str]) -> bool:
    """Check if any pattern matches in text (case-insensitive)."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def analyse_case(gt: str, pred: str) -> dict:
    """Return per-concept status for one case."""
    result = {}
    for concept, patterns in CONCEPTS.items():
        in_gt   = concept_present(gt, patterns)
        in_pred = concept_present(pred, patterns)

        if in_gt and in_pred:
            result[concept] = "match"
        elif in_gt and not in_pred:
            result[concept] = "missed"
        elif not in_gt and in_pred:
            result[concept] = "wrong"
        else:
            result[concept] = "absent"     # neither has it — not relevant
    return result


def run_experiment(exp_name: str, split: str = "test") -> dict:
    """Run semantic error analysis on one experiment."""
    pred_path = Path(f"runs/{exp_name}/predictions_{split}.jsonl")
    if not pred_path.exists():
        print(f"  ❌ Not found: {pred_path}")
        return {}

    with open(pred_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    # Per-case analysis
    all_cases = []
    concept_counts = defaultdict(lambda: {"match": 0, "missed": 0, "wrong": 0, "absent": 0})

    for s in samples:
        case_result = analyse_case(s["gt"], s["pred"])
        case_result["case_id"] = s["case_id"]
        all_cases.append(case_result)

        for concept, status in case_result.items():
            if concept == "case_id":
                continue
            concept_counts[concept][status] += 1

    # ── Global aggregation ───────────────────────────────────────
    total_match  = sum(v["match"]  for v in concept_counts.values())
    total_missed = sum(v["missed"] for v in concept_counts.values())
    total_wrong  = sum(v["wrong"]  for v in concept_counts.values())
    total_relevant = total_match + total_missed  # concepts that ARE in GT

    missed_rate = total_missed / total_relevant if total_relevant else 0
    match_rate  = total_match  / total_relevant if total_relevant else 0
    # Wrong rate: out of all concept slots (cases × concepts)
    total_slots = len(samples) * len(CONCEPTS)
    wrong_rate  = total_wrong / total_slots if total_slots else 0

    result = {
        "experiment": exp_name,
        "split": split,
        "n_cases": len(samples),
        "n_concepts": len(CONCEPTS),
        "global": {
            "match_rate":  round(match_rate, 4),
            "missed_rate": round(missed_rate, 4),
            "wrong_rate":  round(wrong_rate, 4),
            "total_match":  total_match,
            "total_missed": total_missed,
            "total_wrong":  total_wrong,
        },
        "per_concept": {},
        "per_case": all_cases,
    }

    # Per-concept breakdown
    for concept in CONCEPTS:
        c = concept_counts[concept]
        relevant = c["match"] + c["missed"]
        result["per_concept"][concept] = {
            "match": c["match"],
            "missed": c["missed"],
            "wrong": c["wrong"],
            "absent": c["absent"],
            "missed_rate": round(c["missed"] / relevant, 4) if relevant else 0,
            "wrong_rate":  round(c["wrong"] / len(samples), 4),
        }

    return result


def print_results(res: dict):
    """Pretty-print results."""
    g = res["global"]
    print(f"\n{'='*70}")
    print(f"  Experiment: {res['experiment']}  |  Split: {res['split']}  |  Cases: {res['n_cases']}")
    print(f"{'='*70}")
    print(f"  GLOBAL:  Match={g['match_rate']:.1%}  |  Missed={g['missed_rate']:.1%}  |  Wrong={g['wrong_rate']:.1%}")
    print(f"           ({g['total_match']} match, {g['total_missed']} missed, {g['total_wrong']} wrong)")
    print(f"\n  {'Concept':<12} {'Match':>6} {'Missed':>7} {'Wrong':>6} {'Absent':>7}  {'MissRate':>9} {'WrongRate':>10}")
    print(f"  {'-'*62}")
    for concept in CONCEPTS:
        c = res["per_concept"][concept]
        print(f"  {concept:<12} {c['match']:>6} {c['missed']:>7} {c['wrong']:>6} {c['absent']:>7}"
              f"  {c['missed_rate']:>8.1%} {c['wrong_rate']:>10.1%}")

    # Show worst cases (most missed + wrong)
    print(f"\n  Top mismatch cases:")
    scored = []
    for case in res["per_case"]:
        n_missed = sum(1 for k, v in case.items() if k != "case_id" and v == "missed")
        n_wrong  = sum(1 for k, v in case.items() if k != "case_id" and v == "wrong")
        if n_missed + n_wrong > 0:
            scored.append((case["case_id"], n_missed, n_wrong, case))
    scored.sort(key=lambda x: x[1] + x[2], reverse=True)
    for cid, nm, nw, case in scored[:5]:
        missed_list = [k for k, v in case.items() if v == "missed"]
        wrong_list  = [k for k, v in case.items() if v == "wrong"]
        print(f"    {cid}: missed={missed_list}  wrong={wrong_list}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment name (e.g. baseline_structured). "
                             "If omitted, runs both and compares.")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    if args.exp:
        res = run_experiment(args.exp, args.split)
        if res:
            print_results(res)
            # Save
            out_path = Path(f"runs/{args.exp}/semantic_errors_{args.split}.json")
            with open(out_path, "w") as f:
                json.dump({k: v for k, v in res.items() if k != "per_case"}, f, indent=2)
            print(f"\n  ✅ Saved: {out_path}")
    else:
        # Run both and compare
        experiments = [
            ("baseline_structured", "Retrieval (top-1)"),
            ("ours_prompt_semantic", "Ours (T5+Vis+Sem)"),
        ]
        results = []
        for exp_name, label in experiments:
            res = run_experiment(exp_name, args.split)
            if res:
                res["label"] = label
                results.append(res)
                print_results(res)
                # Save
                out_path = Path(f"runs/{exp_name}/semantic_errors_{args.split}.json")
                with open(out_path, "w") as f:
                    json.dump({k: v for k, v in res.items() if k != "per_case"}, f, indent=2)

        # ── Comparison table ─────────────────────────────────────
        if len(results) == 2:
            print(f"\n\n{'='*70}")
            print(f"  COMPARISON: Semantic Error Rates ({args.split} set)")
            print(f"{'='*70}")
            print(f"  {'Method':<25} {'Match↑':>8} {'Missed↓':>9} {'Wrong↓':>8}")
            print(f"  {'-'*52}")
            for r in results:
                g = r["global"]
                print(f"  {r['label']:<25} {g['match_rate']:>7.1%} {g['missed_rate']:>9.1%} {g['wrong_rate']:>8.1%}")

            # Per-concept comparison
            print(f"\n  Per-concept missed rate:")
            print(f"  {'Concept':<12}", end="")
            for r in results:
                print(f" {r['label'][:15]:>16}", end="")
            print()
            print(f"  {'-'*45}")
            for concept in CONCEPTS:
                print(f"  {concept:<12}", end="")
                for r in results:
                    mr = r["per_concept"][concept]["missed_rate"]
                    print(f" {mr:>15.1%}", end="")
                print()

            print(f"\n  Per-concept wrong rate:")
            print(f"  {'Concept':<12}", end="")
            for r in results:
                print(f" {r['label'][:15]:>16}", end="")
            print()
            print(f"  {'-'*45}")
            for concept in CONCEPTS:
                print(f"  {concept:<12}", end="")
                for r in results:
                    wr = r["per_concept"][concept]["wrong_rate"]
                    print(f" {wr:>15.1%}", end="")
                print()


if __name__ == "__main__":
    main()
