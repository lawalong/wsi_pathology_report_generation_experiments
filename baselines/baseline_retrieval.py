import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Baseline-B: Nearest Neighbor Retrieval Baseline

For each test case:
1. Find the most similar case in training set (by WSI embedding cosine similarity)
2. Use that case's diagnosis as prediction

This is a strong baseline that:
- Works well even with small datasets (40 cases)
- Easy to explain in paper
- Sets up comparison for later improvements (prompt/semantic loss)
"""

import json
import numpy as np
from pathlib import Path

# Import shared configuration
from config import SPLITS_FILE, TARGETS_FILE, WSI_DIR

# Output for this baseline
OUTPUT_DIR = Path("runs/baseline_retrieval")


def load_vec(case_id: str) -> np.ndarray:
    """Load WSI embedding for a case."""
    return np.load(WSI_DIR / f"{case_id}.npy").astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float((a * b).sum())


def main():
    # Load splits
    splits = json.loads(SPLITS_FILE.read_text())
    
    # Load targets
    targets = {}
    for line in TARGETS_FILE.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        targets[obj["case_id"]] = obj["target"]
    
    # Filter to cases with both target and WSI embedding
    train_ids = [c for c in splits["train"] if c in targets and (WSI_DIR / f"{c}.npy").exists()]
    val_ids = [c for c in splits["val"] if c in targets and (WSI_DIR / f"{c}.npy").exists()]
    test_ids = [c for c in splits["test"] if c in targets and (WSI_DIR / f"{c}.npy").exists()]
    
    print(f"Train: {len(train_ids)} cases")
    print(f"Val:   {len(val_ids)} cases")
    print(f"Test:  {len(test_ids)} cases")
    print("=" * 60)
    
    # Load training embeddings
    print("Loading training embeddings...")
    train_vecs = {cid: load_vec(cid) for cid in train_ids}
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============ Run retrieval on test set ============
    pred_path = OUTPUT_DIR / "predictions_test.jsonl"
    
    print("\nRunning retrieval on test set...")
    with pred_path.open("w", encoding="utf-8") as f:
        for cid in test_ids:
            q = load_vec(cid)
            
            # Find nearest neighbor in training set
            best_match = None
            best_sim = -1e9
            
            for tid, tv in train_vecs.items():
                sim = cosine_similarity(q, tv)
                if sim > best_sim:
                    best_sim = sim
                    best_match = tid
            
            pred = targets[best_match]
            ref = targets[cid]
            
            # Write prediction
            f.write(json.dumps({
                "case_id": cid,
                "nn_case": best_match,
                "similarity": round(best_sim, 4),
                "pred": pred,
                "gt": ref
            }, ensure_ascii=False) + "\n")
            
            print(f"  {cid} → {best_match} (sim={best_sim:.4f})")
    
    print(f"\n✅ Saved predictions: {pred_path}")
    
    # ============ Also run on validation set ============
    val_pred_path = OUTPUT_DIR / "predictions_val.jsonl"
    
    print("\nRunning retrieval on validation set...")
    with val_pred_path.open("w", encoding="utf-8") as f:
        for cid in val_ids:
            q = load_vec(cid)
            
            best_match = None
            best_sim = -1e9
            
            for tid, tv in train_vecs.items():
                sim = cosine_similarity(q, tv)
                if sim > best_sim:
                    best_sim = sim
                    best_match = tid
            
            pred = targets[best_match]
            ref = targets[cid]
            
            f.write(json.dumps({
                "case_id": cid,
                "nn_case": best_match,
                "similarity": round(best_sim, 4),
                "pred": pred,
                "gt": ref
            }, ensure_ascii=False) + "\n")
            
            print(f"  {cid} → {best_match} (sim={best_sim:.4f})")
    
    print(f"\n✅ Saved predictions: {val_pred_path}")
    
    # ============ Summary ============
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Test predictions: {pred_path}")
    print(f"Val predictions:  {val_pred_path}")
    print(f"\nNext step: Run evaluate_metrics.py to compute ROUGE/BERTScore")


if __name__ == "__main__":
    main()
