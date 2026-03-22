import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Semantic Reranking Baseline

Two-stage retrieval:
  Stage 1: Top-k retrieval by WSI cosine similarity (visual)
  Stage 2: Rerank candidates by text-level semantic similarity

For each test case:
1. Retrieve top-k (k=10) candidates by WSI embedding
2. Build a structured query from the candidate pool (consensus fields)
3. Rerank candidates by text-level similarity using sentence embeddings
4. Pick the best reranked candidate → structured output

Paper framing:
  "We introduce a two-stage retrieve-then-rerank pipeline that first
   identifies morphologically similar cases in visual feature space,
   then refines the selection using text-level semantic similarity."

Requirements:
  pip install sentence-transformers
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

# Import shared configuration
from config import SPLITS_FILE, TEXT_DIR, WSI_DIR

SCHEMA_TEMPLATE = """DIAGNOSIS: {diagnosis}

MICROSCOPIC: {microscopic}

MARGINS: {margins}

LYMPH NODES: {lymph_nodes}

STAGE: {ptnm}"""

# Retrieval pool size for reranking
TOPK_POOL = 10
# Final selection after reranking (we pick top-1 after rerank,
# but also try fusing top-3 after rerank)
RERANK_FINAL_K = [1, 3]

OUTPUT_BASE = Path("runs")


def load_vec(case_id: str) -> np.ndarray:
    return np.load(WSI_DIR / f"{case_id}.npy").astype(np.float32)


def load_text(case_id: str) -> dict:
    p = TEXT_DIR / f"{case_id}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def cosine_sim_np(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float((a * b).sum())


def topk_visual(query_vec, train_data, k):
    sims = []
    for tid, tdata in train_data.items():
        sim = cosine_sim_np(query_vec, tdata['vec'])
        sims.append((tid, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


def fields_to_sentence(fields: dict) -> str:
    """Convert structured fields to a single sentence for embedding."""
    parts = []
    for key in ['diagnosis', 'microscopic', 'margins', 'lymph_nodes', 'ptnm']:
        val = fields.get(key, '')
        if val and val.strip() and val.strip().lower() != 'not available':
            parts.append(val.strip())
    return '. '.join(parts) if parts else ''


def build_consensus_query(neighbours, train_data):
    """
    Build a pseudo-query from the top visual candidates.
    This represents what the visual model 'thinks' the case looks like,
    expressed in text — used as the reranking query.
    """
    all_kw = []
    for cid, _ in neighbours:
        t = train_data[cid]['text']
        all_kw.extend(t.get('keywords', []))

    # Top frequent keywords → query
    freq = Counter(kw.lower() for kw in all_kw)
    top_terms = [kw for kw, _ in freq.most_common(10)]
    return ' '.join(top_terms)


def fuse_topk(neighbours, train_data, k):
    """Same fusion logic as topk_fusion baseline."""
    texts = [(cid, train_data[cid]['text']) for cid, _ in neighbours[:k]]

    diagnosis = ""
    for _, t in texts:
        d = t.get('diagnosis', '')
        if d and d.strip():
            diagnosis = d[:500]
            break

    def first_non_empty(field, max_len=300):
        for _, t in texts:
            val = t.get(field, '')
            if val and val.strip() and val.strip().lower() != 'not available':
                return val[:max_len]
        return "Not available"

    microscopic = first_non_empty('microscopic', 300)
    margins = first_non_empty('margins', 200)
    lymph_nodes = first_non_empty('lymph_nodes', 200)

    ptnm_votes = [t.get('ptnm', '').strip() for _, t in texts if t.get('ptnm', '').strip()]
    ptnm = Counter(ptnm_votes).most_common(1)[0][0] if ptnm_votes else "Not specified"

    kw_set, seen = [], set()
    for _, t in texts:
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


def format_output(fields):
    return SCHEMA_TEMPLATE.format(
        diagnosis=fields.get('diagnosis', 'Not available'),
        microscopic=fields.get('microscopic', 'Not available'),
        margins=fields.get('margins', 'Not available'),
        lymph_nodes=fields.get('lymph_nodes', 'Not available'),
        ptnm=fields.get('ptnm', 'Not specified'),
    )


def format_gt(fields):
    return SCHEMA_TEMPLATE.format(
        diagnosis=fields.get('diagnosis', '')[:500],
        microscopic=fields.get('microscopic', '')[:300],
        margins=fields.get('margins', '')[:200],
        lymph_nodes=fields.get('lymph_nodes', '')[:200],
        ptnm=fields.get('ptnm', '') or 'Not specified',
    )


def main():
    # ── Load sentence model ──────────────────────────────────────
    print("Loading sentence-transformers model (first run downloads ~90 MB)...")
    try:
        from sentence_transformers import SentenceTransformer, util as st_util
    except ImportError:
        print("❌ Please install: pip install sentence-transformers")
        return

    # Use a lightweight clinical / biomedical model if available,
    # otherwise the popular all-MiniLM-L6-v2 (fast & good).
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence model loaded.\n")

    # ── Load data ────────────────────────────────────────────────
    splits = json.loads(SPLITS_FILE.read_text())

    train_ids = [c for c in splits["train"]
                 if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]
    test_ids = [c for c in splits["test"]
                if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]
    val_ids = [c for c in splits["val"]
               if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]

    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
    print("=" * 60)

    print("Loading training data (vectors + text + embeddings)...")
    train_data = {}
    train_sentences = {}
    for cid in train_ids:
        t = load_text(cid)
        train_data[cid] = {'vec': load_vec(cid), 'text': t}
        train_sentences[cid] = fields_to_sentence(t)

    # Pre-encode all training sentences
    print("Encoding training text with sentence-transformers...")
    train_sent_list = [train_sentences[cid] for cid in train_ids]
    train_embs = model.encode(train_sent_list, batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True)
    # Store back in dict
    for i, cid in enumerate(train_ids):
        train_data[cid]['sent_emb'] = train_embs[i]

    # ── Run for each final-k variant ────────────────────────────
    for final_k in RERANK_FINAL_K:
        exp_name = f"rerank_k{final_k}"
        out_dir = OUTPUT_BASE / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Semantic Reranking  →  final top-{final_k}")
        print(f"{'='*60}")

        for split_name, case_ids in [("test", test_ids), ("val", val_ids)]:
            pred_path = out_dir / f"predictions_{split_name}.jsonl"
            print(f"\nProcessing {split_name} ({len(case_ids)} cases)...")

            with pred_path.open("w", encoding="utf-8") as f:
                for cid in case_ids:
                    q_vec = load_vec(cid)
                    gt_text = load_text(cid)

                    # Stage 1: visual top-k
                    visual_topk = topk_visual(q_vec, train_data, TOPK_POOL)

                    # Stage 2: build query from visual pool & rerank
                    query_sentence = build_consensus_query(visual_topk, train_data)
                    q_emb = model.encode([query_sentence], convert_to_numpy=True)[0]

                    scored = []
                    for tid, vis_sim in visual_topk:
                        sent_emb = train_data[tid]['sent_emb']
                        # text-level cosine similarity
                        text_sim = float(np.dot(q_emb, sent_emb)
                                         / (np.linalg.norm(q_emb) * np.linalg.norm(sent_emb) + 1e-8))
                        # Combined score: weighted blend of visual and text similarity
                        combined = 0.5 * vis_sim + 0.5 * text_sim
                        scored.append((tid, vis_sim, text_sim, combined))

                    # Sort by combined score
                    scored.sort(key=lambda x: x[3], reverse=True)

                    # Take final-k after reranking
                    reranked = [(s[0], s[3]) for s in scored[:final_k]]

                    if final_k == 1:
                        best_cid = reranked[0][0]
                        nn_text = train_data[best_cid]['text']
                        pred_fields = {
                            'diagnosis': nn_text.get('diagnosis', 'Not available')[:500],
                            'microscopic': nn_text.get('microscopic', 'Not available')[:300],
                            'margins': nn_text.get('margins', 'Not available')[:200],
                            'lymph_nodes': nn_text.get('lymph_nodes', 'Not available')[:200],
                            'ptnm': nn_text.get('ptnm', 'Not specified') or 'Not specified',
                            'keywords': nn_text.get('keywords', []),
                        }
                    else:
                        pred_fields = fuse_topk(reranked, train_data, final_k)

                    pred = format_output(pred_fields)
                    gt = format_gt(gt_text)
                    gt_keywords = gt_text.get('keywords', [])

                    nn_ids = [s[0] for s in scored[:final_k]]
                    nn_sims = [round(s[3], 4) for s in scored[:final_k]]

                    f.write(json.dumps({
                        "case_id": cid,
                        "nn_cases": nn_ids,
                        "similarities": nn_sims,
                        "nn_case": nn_ids[0],
                        "similarity": nn_sims[0],
                        "visual_top1": scored[0][0],
                        "visual_sim": round(scored[0][1], 4),
                        "text_sim": round(scored[0][2], 4),
                        "pred": pred,
                        "gt": gt,
                        "pred_diagnosis": pred_fields['diagnosis'],
                        "gt_diagnosis": gt_text.get('diagnosis', '')[:500],
                        "gt_keywords": gt_keywords,
                        "pred_keywords": pred_fields.get('keywords', []),
                    }, ensure_ascii=False) + "\n")

                    vis_top = scored[0]
                    rerank_top = scored[0]  # after sort
                    changed = "🔄" if reranked[0][0] != visual_topk[0][0] else "  "
                    print(f"  {changed} {cid} → {nn_ids[0]} "
                          f"(vis={vis_top[1]:.4f} txt={vis_top[2]:.4f} comb={vis_top[3]:.4f})")

            print(f"✅ Saved: {pred_path}")

    print("\n" + "=" * 60)
    print("SEMANTIC RERANKING COMPLETE")
    print("=" * 60)
    print("Next: python evaluate_metrics.py --exp rerank_k1 --split test --bertscore")


if __name__ == "__main__":
    main()
