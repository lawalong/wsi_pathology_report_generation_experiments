import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Baseline with Structured Output: Retrieval + Schema-based Rewriting

For each test case:
1. Find nearest neighbor in training set (by WSI embedding)
2. Extract structured fields from nn_report
3. Output in fixed schema format

This demonstrates "prompt strategy enforcing structure" without LLM.
"""

import json
import numpy as np
from pathlib import Path

# Import shared configuration
from config import SPLITS_FILE, TEXT_DIR, WSI_DIR

# Output for this baseline
OUTPUT_DIR = Path("runs/baseline_structured")

# Output schema template
SCHEMA_TEMPLATE = """DIAGNOSIS: {diagnosis}

MICROSCOPIC: {microscopic}

MARGINS: {margins}

LYMPH NODES: {lymph_nodes}

STAGE: {ptnm}"""


def load_vec(case_id: str) -> np.ndarray:
    """Load WSI embedding for a case."""
    return np.load(WSI_DIR / f"{case_id}.npy").astype(np.float32)


def load_structured_text(case_id: str) -> dict:
    """Load structured text for a case."""
    text_path = TEXT_DIR / f"{case_id}.json"
    if text_path.exists():
        return json.loads(text_path.read_text(encoding="utf-8"))
    return {}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float((a * b).sum())


def format_structured_output(fields: dict) -> str:
    """Format structured fields into schema template."""
    return SCHEMA_TEMPLATE.format(
        diagnosis=fields.get('diagnosis', 'Not available')[:500],
        microscopic=fields.get('microscopic', 'Not available')[:300],
        margins=fields.get('margins', 'Not available')[:200],
        lymph_nodes=fields.get('lymph_nodes', 'Not available')[:200],
        ptnm=fields.get('ptnm', 'Not available') or 'Not specified'
    )


def format_gt_structured(fields: dict) -> str:
    """Format GT in same schema for fair comparison."""
    return SCHEMA_TEMPLATE.format(
        diagnosis=fields.get('diagnosis', '')[:500],
        microscopic=fields.get('microscopic', '')[:300],
        margins=fields.get('margins', '')[:200],
        lymph_nodes=fields.get('lymph_nodes', '')[:200],
        ptnm=fields.get('ptnm', '') or 'Not specified'
    )


def main():
    # Load splits
    splits = json.loads(SPLITS_FILE.read_text())
    
    # Filter to cases with both text and WSI
    train_ids = [c for c in splits["train"] 
                 if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]
    test_ids = [c for c in splits["test"] 
                if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]
    val_ids = [c for c in splits["val"] 
               if (TEXT_DIR / f"{c}.json").exists() and (WSI_DIR / f"{c}.npy").exists()]
    
    print(f"Train: {len(train_ids)} cases")
    print(f"Val:   {len(val_ids)} cases")
    print(f"Test:  {len(test_ids)} cases")
    print("=" * 60)
    
    # Load training embeddings and structured text
    print("Loading training data...")
    train_data = {}
    for cid in train_ids:
        train_data[cid] = {
            'vec': load_vec(cid),
            'text': load_structured_text(cid)
        }
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name, case_ids in [("test", test_ids), ("val", val_ids)]:
        pred_path = OUTPUT_DIR / f"predictions_{split_name}.jsonl"
        
        print(f"\nProcessing {split_name} set...")
        
        with pred_path.open("w", encoding="utf-8") as f:
            for cid in case_ids:
                q_vec = load_vec(cid)
                gt_text = load_structured_text(cid)
                
                # Find nearest neighbor
                best_match = None
                best_sim = -1e9
                
                for tid, tdata in train_data.items():
                    sim = cosine_similarity(q_vec, tdata['vec'])
                    if sim > best_sim:
                        best_sim = sim
                        best_match = tid
                
                # Get NN's structured text
                nn_text = train_data[best_match]['text']
                
                # Format prediction in schema
                pred = format_structured_output(nn_text)
                
                # Format GT in same schema for fair comparison
                gt = format_gt_structured(gt_text)
                
                # Also keep raw diagnosis for comparison
                pred_diagnosis = nn_text.get('diagnosis', '')[:500]
                gt_diagnosis = gt_text.get('diagnosis', '')[:500]
                
                # Keywords for semantic coverage
                gt_keywords = gt_text.get('keywords', [])
                nn_keywords = nn_text.get('keywords', [])
                
                f.write(json.dumps({
                    "case_id": cid,
                    "nn_case": best_match,
                    "similarity": round(best_sim, 4),
                    "pred": pred,
                    "gt": gt,
                    "pred_diagnosis": pred_diagnosis,
                    "gt_diagnosis": gt_diagnosis,
                    "gt_keywords": gt_keywords,
                    "pred_keywords": nn_keywords
                }, ensure_ascii=False) + "\n")
                
                print(f"  {cid} → {best_match} (sim={best_sim:.4f})")
        
        print(f"✅ Saved: {pred_path}")
    
    print("\n" + "=" * 60)
    print("STRUCTURED BASELINE COMPLETE")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print("\nNext: python evaluate_metrics.py --exp baseline_structured --split test")


if __name__ == "__main__":
    main()
