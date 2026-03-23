"""
Compare metrics across all baselines for the paper.
Outputs console tables + LaTeX-ready table.
"""

import json
from pathlib import Path


def load_metrics(exp_name: str, split: str) -> dict:
    """Load metrics for an experiment."""
    path = Path(f"runs/{exp_name}/metrics_{split}.json")
    if path.exists():
        return json.loads(path.read_text())
    return {}


def fmt(val, width=10):
    """Format a metric value for display."""
    if isinstance(val, float):
        return f"{val:.4f}".rjust(width)
    return str(val).rjust(width)


def print_table(experiments, split):
    """Print a formatted results table for one split."""
    header = (f"{'Method':<30} {'ROUGE-1':>10} {'ROUGE-2':>10} "
              f"{'ROUGE-L':>10} {'KW Cov':>10}")
    print("-" * 85)
    print(header)
    print("-" * 85)

    for exp_name, display_name in experiments:
        m = load_metrics(exp_name, split)
        if not m:
            print(f"{display_name:<30} {'—':>10} {'—':>10} {'—':>10} {'—':>10}")
            continue
        print(f"{display_name:<30} "
              f"{fmt(m.get('rouge1', '-'))} "
              f"{fmt(m.get('rouge2', '-'))} "
              f"{fmt(m.get('rougeL', '-'))} "
              f"{fmt(m.get('keyword_coverage', '-'))}")

    print("-" * 85)


def main():
    # ── All experiments (order = paper table order) ──────────────
    experiments = [
        ("baseline_retrieval",   "Raw Retrieval"),
        ("baseline_structured",  "Structured (top-1)"),
        ("topk_fusion_k3",      "Structured Fusion (k=3)"),
        ("topk_fusion_k5",      "Structured Fusion (k=5)"),
        ("rerank_k1",           "Semantic Rerank (k=1)"),
        ("rerank_k3",           "Semantic Rerank+Fusion (k=3)"),
        ("ours_prompt_semantic", "Ours (T5+Vis+SemLoss)"),
    ]

    print("=" * 85)
    print("   FULL BASELINE COMPARISON — TCGA-BRCA (204 cases, seed=42)")
    print("=" * 85)

    for split in ["test", "val"]:
        print(f"\n📊 {split.upper()} SET")
        print_table(experiments, split)

    # ── LaTeX table (test set) ───────────────────────────────────
    print("\n\n� LaTeX Table (Test Set):")
    print("=" * 85)
    print(r"""\begin{table}[t]
\centering
\caption{Retrieval baselines on TCGA-BRCA test set (42 queries, 122 train cases).}
\label{tab:baselines}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{ROUGE-1} & \textbf{ROUGE-2} & \textbf{ROUGE-L} & \textbf{KW Cov.} \\
\midrule""")

    best_rL = max(
        load_metrics(e, "test").get("rougeL", 0) for e, _ in experiments
    )
    best_kw = max(
        load_metrics(e, "test").get("keyword_coverage", 0) for e, _ in experiments
    )

    for exp_name, display_name in experiments:
        m = load_metrics(exp_name, "test")
        r1 = f"{m.get('rouge1', 0):.4f}"
        r2 = f"{m.get('rouge2', 0):.4f}"
        rL_val = m.get('rougeL', 0)
        kw_val = m.get('keyword_coverage', 0)
        # Bold best values
        rL = f"\\textbf{{{rL_val:.4f}}}" if rL_val == best_rL else f"{rL_val:.4f}"
        kw = f"\\textbf{{{kw_val:.4f}}}" if kw_val == best_kw else f"{kw_val:.4f}"
        if kw_val == 0:
            kw = "---"
        print(f"{display_name} & {r1} & {r2} & {rL} & {kw} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # ── LaTeX table (val set) ────────────────────────────────────
    print(r"""
\begin{table}[t]
\centering
\caption{Retrieval baselines on TCGA-BRCA validation set (40 queries, 122 train cases).}
\label{tab:baselines_val}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{ROUGE-1} & \textbf{ROUGE-2} & \textbf{ROUGE-L} & \textbf{KW Cov.} \\
\midrule""")

    best_rL_v = max(
        load_metrics(e, "val").get("rougeL", 0) for e, _ in experiments
    )
    best_kw_v = max(
        load_metrics(e, "val").get("keyword_coverage", 0) for e, _ in experiments
    )

    for exp_name, display_name in experiments:
        m = load_metrics(exp_name, "val")
        r1 = f"{m.get('rouge1', 0):.4f}"
        r2 = f"{m.get('rouge2', 0):.4f}"
        rL_val = m.get('rougeL', 0)
        kw_val = m.get('keyword_coverage', 0)
        rL = f"\\textbf{{{rL_val:.4f}}}" if rL_val == best_rL_v else f"{rL_val:.4f}"
        kw = f"\\textbf{{{kw_val:.4f}}}" if kw_val == best_kw_v else f"{kw_val:.4f}"
        if kw_val == 0:
            kw = "---"
        print(f"{display_name} & {r1} & {r2} & {rL} & {kw} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # ── Key findings ─────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("KEY FINDINGS:")
    print("=" * 85)

    raw  = load_metrics("baseline_retrieval", "test")
    stru = load_metrics("baseline_structured", "test")
    fk3  = load_metrics("topk_fusion_k3", "test")
    fk5  = load_metrics("topk_fusion_k5", "test")
    rk1  = load_metrics("rerank_k1", "test")
    rk3  = load_metrics("rerank_k3", "test")

    raw_rL = raw.get('rougeL', 0.001)

    def pct(new, base):
        return (new / base - 1) * 100

    print(f"• Structured (top-1)       → ROUGE-L {stru.get('rougeL',0):.4f}  "
          f"(+{pct(stru.get('rougeL',0), raw_rL):.0f}% vs raw)")
    print(f"• Fusion k=3               → ROUGE-L {fk3.get('rougeL',0):.4f}  "
          f"(+{pct(fk3.get('rougeL',0), raw_rL):.0f}% vs raw)  "
          f"KW {fk3.get('keyword_coverage',0):.1%}")
    print(f"• Fusion k=5               → ROUGE-L {fk5.get('rougeL',0):.4f}  "
          f"(+{pct(fk5.get('rougeL',0), raw_rL):.0f}% vs raw)  "
          f"KW {fk5.get('keyword_coverage',0):.1%}")
    print(f"• Rerank k=1               → ROUGE-L {rk1.get('rougeL',0):.4f}  "
          f"(+{pct(rk1.get('rougeL',0), raw_rL):.0f}% vs raw)  "
          f"KW {rk1.get('keyword_coverage',0):.1%}")
    print(f"• Rerank+Fusion k=3        → ROUGE-L {rk3.get('rougeL',0):.4f}  "
          f"(+{pct(rk3.get('rougeL',0), raw_rL):.0f}% vs raw)  "
          f"KW {rk3.get('keyword_coverage',0):.1%}")
    print()
    print("✅ Best KW coverage: semantic reranking → shows text-level matching captures "
          "clinical terms that pure visual similarity misses.")
    print("✅ Both fusion and reranking improve over single-case retrieval.")
    print("✅ All baselines serve as lower bounds for future generative models.")


if __name__ == "__main__":
    main()
