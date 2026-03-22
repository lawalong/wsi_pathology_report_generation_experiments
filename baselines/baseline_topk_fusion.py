import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Top-k Structured Fusion Baseline

Instead of relying on a single retrieved case (top-1), we aggregate
multiple similar cases (top-k) to improve robustness.

For each test case:
1. Retrieve top-k most similar training cases (by WSI embedding)
2. Aggregate structured fields across the k candidates:
   - Diagnosis: pick from the nearest neighbour (most reliable)
   - Microscopic/Margins/Lymph_nodes: merge non-empty fields,
     preferring the nearest neighbour but filling gaps from others
   - Keywords: union of all k candidates' keywords
   - pTNM: majority vote among candidates
3. Output in the same schema template

Paper framing:
  "Instead of relying on a single retrieved case, we aggregate multiple
   similar cases to improve robustness and field coverage."
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

# Import shared configuration
from config import SPLITS_FILE, TEXT_DIR, WSI_DIR

# Schema (same as baseline_structured)
SCHEMA_TEMPLATE = """DIAGNOSIS: {diagnosis}

MICROSCOPIC: {microscopic}

MARGINS: {margins}

LYMPH NODES: {lymph_nodes}

STAGE: {ptnm}"""

K_VALUES = [3, 5]  # We'll generate predictions for each k


def load_vec(case_id: str) -> np.ndarray:
    return np.load(WSI_DIR / f"{case_id}.npy").astype(np.float32)


def load_text(case_id: str) -> dict:
    p = TEXT_DIR / f"{case_id}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float((a * b).sum())


def topk_retrieval(query_vec, train_data, k):
    """Return top-k (case_id, similarity) pairs sorted by descending similarity."""
    sims = []
    for tid, tdata in train_data.items():
        sim = cosine_similarity(query_vec, tdata['vec'])
        sims.append((tid, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


def fuse_fields(neighbours: list, train_data: dict) -> dict:
    """
    Aggregate structured fields from k neighbours.
    
    Strategy:
    - diagnosis: always take from top-1 (most morphologically similar)
    - microscopic, margins, lymph_nodes: take from top-1 if non-empty,
      otherwise fill from the next closest neighbour that has it
    - ptnm: majority vote across all k
    - keywords: union of all k neighbours' keywords (deduplicated)
    """
    # neighbours is list of (case_id, sim)
    texts = [(cid, train_data[cid]['text']) for cid, _ in neighbours]

    # ---------- diagnosis: top-1 ----------
    diagnosis = ""
    for cid, t in texts:
        d = t.get('diagnosis', '')
        if d and d.strip():
            diagnosis = d[:500]
            break

    # ---------- fill-from-nearest for other fields ----------
    def first_non_empty(field, max_len=300):
        for cid, t in texts:
            val = t.get(field, '')
            if val and val.strip() and val.strip().lower() != 'not available':
                return val[:max_len]
        return "Not available"

    microscopic = first_non_empty('microscopic', 300)
    margins = first_non_empty('margins', 200)
    lymph_nodes = first_non_empty('lymph_nodes', 200)

    # ---------- ptnm: majority vote ----------
    ptnm_votes = []
    for cid, t in texts:
        p = t.get('ptnm', '')
        if p and p.strip():
            ptnm_votes.append(p.strip())
    if ptnm_votes:
        ptnm = Counter(ptnm_votes).most_common(1)[0][0]
    else:
        ptnm = "Not specified"

    # ---------- keywords: union ----------
    kw_set = []
    seen = set()
    for cid, t in texts:
        for kw in t.get('keywords', []):
            kl = kw.lower()
            if kl not in seen:
                seen.add(kl)
                kw_set.append(kw)

    return {
        'diagnosis': diagnosis,
        'microscopic': microscopic,
        'margins': margins,
        'lymph_nodes': lymph_nodes,
        'ptnm': ptnm,
        'keywords': kw_set,
    }


def format_output(fields: dict) -> str:
    return SCHEMA_TEMPLATE.format(
        diagnosis=fields.get('diagnosis', 'Not available'),
        microscopic=fields.get('microscopic', 'Not available'),
        margins=fields.get('margins', 'Not available'),
        lymph_nodes=fields.get('lymph_nodes', 'Not available'),
        ptnm=fields.get('ptnm', 'Not specified'),
    )


def format_gt(fields: dict) -> str:
    return SCHEMA_TEMPLATE.format(
        diagnosis=fields.get('diagnosis', '')[:500],
        microscopic=fields.get('microscopic', '')[:300],
        margins=fields.get('margins', '')[:200],
        lymph_nodes=fields.get('lymph_nodes', '')[:200],
        ptnm=fields.get('ptnm', '') or 'Not specified',
    )


def main():
    # Load splits
    splits = json.loads(SPLITS_FILE.read_text())

    train_ids = [c for c in splits["train"]
                 if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]
    test_ids = [c for c in splits["test"]
                if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]
    val_ids = [c for c in splits["val"]
               if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]

    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
    print("=" * 60)

    # Pre-load training data
    print("Loading training data...")
    train_data = {}
    for cid in train_ids:
        train_data[cid] = {'vec': load_vec(cid), 'text': load_text(cid)}

    # Run for each k
    for k in K_VALUES:
        exp_name = f"topk_fusion_k{k}"
        out_dir = Path(f"runs/{exp_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Top-{k} Structured Fusion")
        print(f"{'='*60}")

        for split_name, case_ids in [("test", test_ids), ("val", val_ids)]:
            pred_path = out_dir / f"predictions_{split_name}.jsonl"
            print(f"\nProcessing {split_name} ({len(case_ids)} cases)...")

            with pred_path.open("w", encoding="utf-8") as f:
                for cid in case_ids:
                    q_vec = load_vec(cid)
                    gt_text = load_text(cid)

                    neighbours = topk_retrieval(q_vec, train_data, k)
                    fused = fuse_fields(neighbours, train_data)

                    pred = format_output(fused)
                    gt = format_gt(gt_text)

                    gt_keywords = gt_text.get('keywords', [])

                    nn_ids = [n[0] for n in neighbours]
                    nn_sims = [round(n[1], 4) for n in neighbours]

                    f.write(json.dumps({
                        "case_id": cid,
                        "nn_cases": nn_ids,
                        "similarities": nn_sims,
                        "nn_case": nn_ids[0],          # compat with evaluate_metrics
                        "similarity": nn_sims[0],
                        "pred": pred,
                        "gt": gt,
                        "pred_diagnosis": fused['diagnosis'],
                        "gt_diagnosis": gt_text.get('diagnosis', '')[:500],
                        "gt_keywords": gt_keywords,
                        "pred_keywords": fused['keywords'],
                    }, ensure_ascii=False) + "\n")

                    print(f"  {cid} → [{', '.join(nn_ids)}] (sims={nn_sims})")

            print(f"✅ Saved: {pred_path}")

    print("\n" + "=" * 60)
    print("TOP-K FUSION COMPLETE")
    print("=" * 60)
    print("Next: python evaluate_metrics.py --exp topk_fusion_k3 --split test --bertscore")


if __name__ == "__main__":
    main()
