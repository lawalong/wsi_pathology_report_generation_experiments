import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Oracle sanity check: Use GT as prediction to verify evaluation pipeline.
Expected: ROUGE scores should be ~1.0 (or very close).

If not close to 1.0, there's a problem with:
- GT text loading/cleaning
- Text preprocessing differences
- OCR noise or formatting issues
"""

import json
from pathlib import Path

# Import shared configuration
from config import SPLITS_FILE, TARGETS_FILE

ORACLE_OUTPUT_DIR = Path("runs/oracle_check")


def main():
    # Load splits
    splits = json.loads(SPLITS_FILE.read_text())
    
    # Load targets
    targets = {}
    for line in TARGETS_FILE.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        targets[obj["case_id"]] = obj["target"]
    
    test_ids = [c for c in splits["test"] if c in targets]
    val_ids = [c for c in splits["val"] if c in targets]
    
    print(f"Test cases: {len(test_ids)}")
    print(f"Val cases:  {len(val_ids)}")
    print("=" * 60)
    
    # Create output directory
    ORACLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Oracle predictions: pred = gt (should give ROUGE ~1.0)
    for split_name, case_ids in [("test", test_ids), ("val", val_ids)]:
        pred_path = ORACLE_OUTPUT_DIR / f"predictions_{split_name}.jsonl"
        
        with pred_path.open("w", encoding="utf-8") as f:
            for cid in case_ids:
                gt = targets[cid]
                # Oracle: prediction = ground truth
                f.write(json.dumps({
                    "case_id": cid,
                    "nn_case": cid,  # Self-retrieval
                    "similarity": 1.0,
                    "pred": gt,      # Prediction = GT
                    "gt": gt
                }, ensure_ascii=False) + "\n")
        
        print(f"✅ Saved oracle predictions: {pred_path}")
    
    print("\n" + "=" * 60)
    print("Now run: python evaluate_metrics.py --exp oracle_check --split test")
    print("Expected: ROUGE-1, ROUGE-2, ROUGE-L should all be ~1.0")
    print("\nIf not 1.0, check GT text loading/preprocessing.")


if __name__ == "__main__":
    main()
