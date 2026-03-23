"""
Semantic Ablation — Controlled comparison across semantic loss weights.

Orchestrates training + prediction + evaluation for multiple lambda values.
Everything is identical across runs except the semantic loss weight.

Usage:
  python3 run_semantic_ablation.py --weights 0.0,0.2 --do_train --do_predict --do_eval
  python3 run_semantic_ablation.py --weights 0.0,0.1,0.2,0.5 --do_train --do_predict --do_eval
  python3 run_semantic_ablation.py --weights 0.0,0.2 --do_eval --splits test  # eval only

Output:
  runs/ours_semantic_l0/    (lambda=0.0, no semantic loss)
  runs/ours_semantic_l01/   (lambda=0.1)
  runs/ours_semantic_l02/   (lambda=0.2, current main method)
  runs/ours_semantic_l05/   (lambda=0.5)
  results/semantic_ablation_summary.{csv,json}
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def weight_to_exp_name(w: float) -> str:
    """Convert weight float to clean experiment name.
    0.0  -> ours_semantic_l0
    0.1  -> ours_semantic_l01
    0.2  -> ours_semantic_l02
    0.5  -> ours_semantic_l05
    1.0  -> ours_semantic_l10
    """
    # Format: remove the decimal point from the string representation
    # 0.0 -> "00" -> "0", 0.1 -> "01", 0.2 -> "02", 0.5 -> "05", 1.0 -> "10"
    s = f"{w:.1f}".replace(".", "")   # "00", "01", "02", "05", "10"
    if s == "00":
        s = "0"
    return f"ours_semantic_l{s}"


def run_cmd(cmd: list[str], label: str):
    """Run a subprocess command with clear error handling."""
    print(f"\n{'─'*60}")
    print(f"  ▶ {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*60}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {label}")
        print(f"   Return code: {result.returncode}")
        sys.exit(1)


def collect_summary(exp_names: list[str], weights: list[float],
                    splits: list[str]) -> list[dict]:
    """Collect metrics from all runs into a summary table."""
    rows = []

    for exp_name, w in zip(exp_names, weights):
        row = {
            "exp_name": exp_name,
            "semantic_weight": w,
        }

        # Read metrics for each split
        for split in splits:
            metrics_path = Path(f"runs/{exp_name}/metrics_{split}.json")
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                row[f"{split}_rouge1"] = m.get("rouge1", None)
                row[f"{split}_rouge2"] = m.get("rouge2", None)
                row[f"{split}_rougeL"] = m.get("rougeL", None)
                row[f"{split}_keyword_coverage"] = m.get("keyword_coverage", None)
            else:
                print(f"  ⚠ Missing: {metrics_path}")

        # Read semantic errors (test only typically)
        sem_path = Path(f"runs/{exp_name}/semantic_errors_test.json")
        if sem_path.exists():
            with open(sem_path) as f:
                se = json.load(f)
            g = se.get("global", {})
            row["test_semantic_match"] = g.get("match_rate", None)
            row["test_semantic_missed"] = g.get("missed_rate", None)
            row["test_semantic_wrong"] = g.get("wrong_rate", None)
        else:
            # Only warn if test split was requested
            if "test" in splits:
                print(f"  ⚠ Missing: {sem_path}")

        rows.append(row)

    return rows


def save_summary(rows: list[dict], out_dir: Path):
    """Save summary as CSV and JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "semantic_ablation_summary.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  ✅ {json_path}")

    # CSV
    csv_path = out_dir / "semantic_ablation_summary.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                vals = []
                for k in keys:
                    v = row.get(k)
                    vals.append("" if v is None else str(v))
                f.write(",".join(vals) + "\n")
    print(f"  ✅ {csv_path}")


def print_summary_table(rows: list[dict]):
    """Print ASCII summary table."""
    print(f"\n{'='*80}")
    print(f"  SEMANTIC ABLATION SUMMARY")
    print(f"{'='*80}")

    # Header
    print(f"  {'Exp':<22} {'λ':>4} {'Val R-L':>8} {'Test R-L':>9} "
          f"{'Val KW':>7} {'Test KW':>8} {'Wrong↓':>7}")
    print(f"  {'─'*74}")

    for row in rows:
        val_rl = row.get("val_rougeL")
        test_rl = row.get("test_rougeL")
        val_kw = row.get("val_keyword_coverage")
        test_kw = row.get("test_keyword_coverage")
        wrong = row.get("test_semantic_wrong")

        val_rl_s = f"{val_rl:.4f}" if val_rl is not None else "—"
        test_rl_s = f"{test_rl:.4f}" if test_rl is not None else "—"
        val_kw_s = f"{val_kw:.4f}" if val_kw is not None else "—"
        test_kw_s = f"{test_kw:.4f}" if test_kw is not None else "—"
        wrong_s = f"{wrong:.1%}" if wrong is not None else "—"

        print(f"  {row['exp_name']:<22} {row['semantic_weight']:>4.1f} "
              f"{val_rl_s:>8} {test_rl_s:>9} {val_kw_s:>7} {test_kw_s:>8} {wrong_s:>7}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Semantic ablation: controlled comparison across lambda values")
    parser.add_argument("--weights", type=str, default="0.0,0.2",
                        help="Comma-separated semantic loss weights (default: 0.0,0.2)")
    parser.add_argument("--do_train", action="store_true",
                        help="Run training for each weight")
    parser.add_argument("--do_predict", action="store_true",
                        help="Run prediction for each weight")
    parser.add_argument("--do_eval", action="store_true",
                        help="Run evaluation for each weight")
    parser.add_argument("--splits", type=str, default="val,test",
                        help="Comma-separated splits for prediction/eval (default: val,test)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    weights = [float(w.strip()) for w in args.weights.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    exp_names = [weight_to_exp_name(w) for w in weights]

    print("=" * 60)
    print("  SEMANTIC ABLATION")
    print("=" * 60)
    print(f"  Weights:  {weights}")
    print(f"  Exp names: {exp_names}")
    print(f"  Splits:   {splits}")
    print(f"  Train:    {args.do_train}")
    print(f"  Predict:  {args.do_predict}")
    print(f"  Eval:     {args.do_eval}")
    if args.dry_run:
        print(f"  *** DRY RUN — no commands will be executed ***")
    print()

    python = sys.executable  # use same python as this script

    for i, (w, exp_name) in enumerate(zip(weights, exp_names)):
        print(f"\n{'═'*60}")
        print(f"  [{i+1}/{len(weights)}] λ = {w}  →  {exp_name}")
        print(f"{'═'*60}")

        # ── Train ───────────────────────────────────────────────
        if args.do_train:
            cmd = [python, "train_prompt_semantic.py",
                   "--exp_name", exp_name,
                   "--semantic_weight", str(w)]
            if args.dry_run:
                print(f"  [DRY] {' '.join(cmd)}")
            else:
                run_cmd(cmd, f"Train {exp_name} (λ={w})")

        # ── Predict ──────────────────────────────────────────────
        if args.do_predict:
            for split in splits:
                cmd = [python, "predict_prompt_semantic.py",
                       "--exp_name", exp_name,
                       "--split", split]
                if args.dry_run:
                    print(f"  [DRY] {' '.join(cmd)}")
                else:
                    run_cmd(cmd, f"Predict {exp_name} — {split}")

        # ── Evaluate ─────────────────────────────────────────────
        if args.do_eval:
            for split in splits:
                cmd = [python, "evaluate_metrics.py",
                       "--exp", exp_name,
                       "--split", split]
                if args.dry_run:
                    print(f"  [DRY] {' '.join(cmd)}")
                else:
                    run_cmd(cmd, f"Evaluate {exp_name} — {split}")

            # Semantic errors (test only)
            if "test" in splits:
                cmd = [python, "compute_semantic_errors.py",
                       "--exp", exp_name,
                       "--split", "test"]
                if args.dry_run:
                    print(f"  [DRY] {' '.join(cmd)}")
                else:
                    run_cmd(cmd, f"Semantic errors {exp_name} — test")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  COLLECTING SUMMARY")
    print(f"{'═'*60}")

    rows = collect_summary(exp_names, weights, splits)
    print_summary_table(rows)

    if not args.dry_run:
        save_summary(rows, Path("results"))

    print("✅ Semantic ablation complete.")


if __name__ == "__main__":
    main()
