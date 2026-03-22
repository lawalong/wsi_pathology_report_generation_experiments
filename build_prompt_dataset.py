"""
Build prompt dataset for training the generative model.

For each case, creates a (prompt_text, target_text) pair:
- prompt_text: structured hints from retrieval + keywords
- target_text: ground-truth structured report

Train split: cross-retrieval within train set (leave-one-out top-k)
Val/Test splits: use existing topk_fusion_k3 predictions as hints

Output:
  data/prompt_train.jsonl
  data/prompt_val.jsonl
  data/prompt_test.jsonl
"""

import json
import numpy as np
from pathlib import Path

from config import SPLITS_FILE, TEXT_DIR, WSI_DIR, OUTPUT_DIR

TOPK_PRED_DIR = Path("runs/topk_fusion_k3")
K = 3  # same k as topk_fusion_k3

SCHEMA_TEMPLATE = """DIAGNOSIS: {diagnosis}

MICROSCOPIC: {microscopic}

MARGINS: {margins}

LYMPH NODES: {lymph_nodes}

STAGE: {ptnm}"""


def load_vec(case_id):
    return np.load(WSI_DIR / f"{case_id}.npy").astype(np.float32)


def load_text(case_id):
    p = TEXT_DIR / f"{case_id}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float((a * b).sum())


def build_prompt(hint_fields: dict) -> str:
    """Build the text prompt from structured hint fields."""
    parts = [
        "Generate a structured pathology report. "
        "You MUST include all clinical terms listed below and follow the EXACT output structure.",
        "",
    ]
    parts.append(f"Similar case diagnosis: {hint_fields.get('diagnosis', 'N/A')[:400]}")
    parts.append(f"Microscopic hint: {hint_fields.get('microscopic', 'N/A')[:250]}")
    parts.append(f"Margins hint: {hint_fields.get('margins', 'N/A')[:150]}")
    parts.append(f"Lymph nodes hint: {hint_fields.get('lymph_nodes', 'N/A')[:150]}")
    parts.append(f"Stage hint: {hint_fields.get('ptnm', 'N/A')}")

    # ── Keyword emphasis (强化关键词) ────────────────────────────
    kws = hint_fields.get('keywords', [])
    if kws:
        parts.append("")
        parts.append("IMPORTANT CLINICAL TERMS (must include in report):")
        for kw in kws[:15]:
            parts.append(f"- {kw}")

    # ── Structural constraint (强制输出结构) ──────────────────────
    parts.append("")
    parts.append("Follow EXACT output structure:")
    parts.append("DIAGNOSIS: <diagnosis text>")
    parts.append("MICROSCOPIC: <microscopic description>")
    parts.append("MARGINS: <margin status>")
    parts.append("LYMPH NODES: <lymph node findings>")
    parts.append("STAGE: <pTNM staging>")

    return "\n".join(parts)


def build_target(gt_fields: dict) -> str:
    """Build the target text from GT structured fields."""
    return SCHEMA_TEMPLATE.format(
        diagnosis=gt_fields.get('diagnosis', '')[:500],
        microscopic=gt_fields.get('microscopic', '')[:300],
        margins=gt_fields.get('margins', '')[:200],
        lymph_nodes=gt_fields.get('lymph_nodes', '')[:200],
        ptnm=gt_fields.get('ptnm', '') or 'Not specified',
    )


def fuse_topk_fields(neighbours_texts):
    """Fuse structured fields from k neighbours (same logic as topk_fusion)."""
    diagnosis = ""
    for t in neighbours_texts:
        d = t.get('diagnosis', '')
        if d and d.strip():
            diagnosis = d[:500]
            break

    def first_non_empty(field, max_len=300):
        for t in neighbours_texts:
            val = t.get(field, '')
            if val and val.strip() and val.strip().lower() != 'not available':
                return val[:max_len]
        return "N/A"

    from collections import Counter
    ptnm_votes = [t.get('ptnm', '').strip()
                   for t in neighbours_texts if t.get('ptnm', '').strip()]
    ptnm = Counter(ptnm_votes).most_common(1)[0][0] if ptnm_votes else "N/A"

    kw_set, seen = [], set()
    for t in neighbours_texts:
        for kw in t.get('keywords', []):
            kl = kw.lower()
            if kl not in seen:
                seen.add(kl)
                kw_set.append(kw)

    return {
        'diagnosis': diagnosis,
        'microscopic': first_non_empty('microscopic', 300),
        'margins': first_non_empty('margins', 200),
        'lymph_nodes': first_non_empty('lymph_nodes', 200),
        'ptnm': ptnm,
        'keywords': kw_set,
    }


def build_train_prompts(train_ids):
    """
    For each train case, do leave-one-out top-k retrieval within train set.
    This gives each training sample a realistic hint (not from itself).
    """
    print("  Loading train embeddings...")
    train_vecs = {cid: load_vec(cid) for cid in train_ids}
    train_texts = {cid: load_text(cid) for cid in train_ids}

    samples = []
    for cid in train_ids:
        q = train_vecs[cid]
        gt = train_texts[cid]
        if not gt:
            continue

        # Find top-k neighbours (excluding self)
        sims = []
        for tid in train_ids:
            if tid == cid:
                continue
            sims.append((tid, cosine_sim(q, train_vecs[tid])))
        sims.sort(key=lambda x: x[1], reverse=True)
        topk = sims[:K]

        nn_texts = [train_texts[t[0]] for t in topk]
        fused = fuse_topk_fields(nn_texts)

        prompt_text = build_prompt(fused)
        target_text = build_target(gt)

        samples.append({
            "case_id": cid,
            "wsi_path": f"data/wsi/{cid}.npy",
            "prompt_text": prompt_text,
            "target_text": target_text,
        })

    return samples


def build_eval_prompts(case_ids, pred_file):
    """
    For val/test, use existing topk_fusion_k3 predictions as hints.
    """
    # Load predictions
    preds = {}
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            preds[obj["case_id"]] = obj

    samples = []
    for cid in case_ids:
        gt = load_text(cid)
        if not gt:
            continue

        if cid in preds:
            # Parse the pred text back into fields for the prompt
            pred_obj = preds[cid]
            # Use pred_keywords and the structured pred text
            hint_fields = {
                'diagnosis': pred_obj.get('pred_diagnosis', ''),
                'microscopic': 'N/A',
                'margins': 'N/A',
                'lymph_nodes': 'N/A',
                'ptnm': 'N/A',
                'keywords': pred_obj.get('pred_keywords', []),
            }
            # Parse structured fields from pred text
            pred_text = pred_obj.get('pred', '')
            for section in ['DIAGNOSIS', 'MICROSCOPIC', 'MARGINS', 'LYMPH NODES', 'STAGE']:
                key = section.lower().replace(' ', '_')
                if key == 'lymph_nodes':
                    marker = 'LYMPH NODES:'
                elif key == 'stage':
                    marker = 'STAGE:'
                    key = 'ptnm'
                else:
                    marker = f"{section}:"
                if marker in pred_text:
                    start = pred_text.index(marker) + len(marker)
                    # Find next section or end
                    rest = pred_text[start:]
                    end = len(rest)
                    for next_sec in ['DIAGNOSIS:', 'MICROSCOPIC:', 'MARGINS:',
                                     'LYMPH NODES:', 'STAGE:']:
                        if next_sec != marker and next_sec in rest:
                            idx = rest.index(next_sec)
                            if idx < end:
                                end = idx
                    hint_fields[key] = rest[:end].strip()
        else:
            # Fallback: use GT text as hint (shouldn't happen for val/test)
            hint_fields = gt

        prompt_text = build_prompt(hint_fields)
        target_text = build_target(gt)

        samples.append({
            "case_id": cid,
            "wsi_path": f"data/wsi/{cid}.npy",
            "prompt_text": prompt_text,
            "target_text": target_text,
        })

    return samples


def write_jsonl(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  ✅ Saved {len(samples)} samples → {path}")


def main():
    splits = json.loads(SPLITS_FILE.read_text())

    # Filter to cases with both text and WSI
    train_ids = [c for c in splits["train"]
                 if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]
    val_ids = [c for c in splits["val"]
               if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]
    test_ids = [c for c in splits["test"]
                if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]

    print(f"Cases — Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
    print("=" * 60)

    # ── Train: cross-retrieval within train set ──────────────────
    print("\n[1/3] Building train prompts (leave-one-out top-k)...")
    train_samples = build_train_prompts(train_ids)
    write_jsonl(train_samples, OUTPUT_DIR / "prompt_train.jsonl")

    # ── Val: from topk_fusion_k3 predictions ─────────────────────
    print("\n[2/3] Building val prompts...")
    val_pred = TOPK_PRED_DIR / "predictions_val.jsonl"
    val_samples = build_eval_prompts(val_ids, val_pred)
    write_jsonl(val_samples, OUTPUT_DIR / "prompt_val.jsonl")

    # ── Test: from topk_fusion_k3 predictions ────────────────────
    print("\n[3/3] Building test prompts...")
    test_pred = TOPK_PRED_DIR / "predictions_test.jsonl"
    test_samples = build_eval_prompts(test_ids, test_pred)
    write_jsonl(test_samples, OUTPUT_DIR / "prompt_test.jsonl")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROMPT DATASET COMPLETE")
    print("=" * 60)
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")

    # Show one example
    if train_samples:
        ex = train_samples[0]
        print(f"\n── Example (train) ──")
        print(f"case_id: {ex['case_id']}")
        print(f"prompt_text ({len(ex['prompt_text'])} chars):")
        print(ex['prompt_text'][:300] + "...")
        print(f"\ntarget_text ({len(ex['target_text'])} chars):")
        print(ex['target_text'][:200] + "...")


if __name__ == "__main__":
    main()
