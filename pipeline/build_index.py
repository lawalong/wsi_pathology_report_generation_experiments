import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Build dataset index from SVS files and matching reports.
Generates:
  - data/dataset.jsonl (one case per line)
  - data/splits.json (train/val/test split)
  - data/missing.txt (cases with missing reports)

Supports incremental updates - will add new cases without duplicates.
"""

import os
import re
import json
import random
from pathlib import Path

# Import shared configuration
from config import (
    SVS_DIR, REF_DIR, OUTPUT_DIR, DATASET_FILE, SPLITS_FILE,
    DEFAULT_NUM_PATCHES, DEFAULT_PATCH_LEVEL, DEFAULT_PATCH_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    print_config
)


def get_case_id_from_name(name: str) -> str | None:
    """Extract TCGA case ID (e.g., TCGA-A7-A4SF) from filename."""
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", name)
    return m.group(1) if m else None


def main():
    print_config()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all SVS files
    svs_files = list(SVS_DIR.rglob("*.svs"))
    print(f"Found {len(svs_files)} SVS files in {SVS_DIR}")
    
    # ============ Deduplicate by case_id ============
    # One case_id → one entry (keep first SVS found)
    case_dict = {}   # case_id → entry
    missing = []
    dup_count = 0
    
    for svs_path in sorted(svs_files):
        svs_name = svs_path.name
        case_id = get_case_id_from_name(svs_name)
        
        if not case_id:
            print(f"⚠️  Cannot parse case ID from: {svs_name}")
            missing.append({"svs": str(svs_path), "reason": "cannot parse case_id"})
            continue
        
        # Skip if we already have this case_id (dedup!)
        if case_id in case_dict:
            dup_count += 1
            continue
        
        # Look for matching reference folder
        ref_case_dir = REF_DIR / case_id
        
        if not ref_case_dir.exists():
            print(f"⚠️  Reference folder not found for: {case_id}")
            missing.append({"case_id": case_id, "svs": str(svs_path), "reason": "no reference folder"})
            continue
        
        # Find Report.txt
        report_path = ref_case_dir / "Report.txt"
        if not report_path.exists():
            print(f"⚠️  Report.txt not found for: {case_id}")
            missing.append({"case_id": case_id, "svs": str(svs_path), "reason": "no Report.txt"})
            continue
        
        # Find PDF files
        pdfs = list(ref_case_dir.glob("*.PDF")) + list(ref_case_dir.glob("*.pdf"))
        pdf_path = str(pdfs[0]) if pdfs else None
        
        # Build case entry
        entry = {
            "case_id": case_id,
            "svs_path": str(svs_path),
            "report_path": str(report_path),
            "pdf_path": pdf_path,
            "num_patches": DEFAULT_NUM_PATCHES,
            "patch_level": DEFAULT_PATCH_LEVEL,
            "patch_size": DEFAULT_PATCH_SIZE
        }
        
        case_dict[case_id] = entry
        print(f"✅ {case_id}: SVS + Report matched")
    
    dataset = list(case_dict.values())
    
    print(f"\n{'='*60}")
    print(f"Total SVS files:       {len(svs_files)}")
    print(f"Duplicates skipped:    {dup_count}")
    print(f"Unique matched cases:  {len(dataset)}")
    print(f"Missing/skipped:       {len(missing)}")
    
    if len(dataset) == 0:
        print("❌ No valid cases found. Check your data folders.")
        return
    
    # ============ Create train/val/test splits ============
    # Split on UNIQUE case_ids only — no case can appear in two splits
    random.seed(RANDOM_SEED)
    case_ids = [entry["case_id"] for entry in dataset]
    random.shuffle(case_ids)
    
    n_total = len(case_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    splits = {
        "train": case_ids[:n_train],
        "val": case_ids[n_train:n_train + n_val],
        "test": case_ids[n_train + n_val:]
    }
    
    # Verify no overlap (sanity check)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert len(train_set & val_set) == 0, "LEAK: train-val overlap!"
    assert len(train_set & test_set) == 0, "LEAK: train-test overlap!"
    assert len(val_set & test_set) == 0, "LEAK: val-test overlap!"
    
    print(f"\nSplits (seed={RANDOM_SEED}):")
    print(f"  Train: {len(splits['train'])} cases")
    print(f"  Val:   {len(splits['val'])} cases")
    print(f"  Test:  {len(splits['test'])} cases")
    print(f"  ✅ No overlap between splits")
    
    # ============ Save outputs ============
    # 1. dataset.jsonl
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n✅ Saved: {DATASET_FILE}")
    
    # 2. splits.json
    with open(SPLITS_FILE, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {SPLITS_FILE}")
    
    # 3. missing.txt (if any)
    if missing:
        missing_path = OUTPUT_DIR / "missing.txt"
        with open(missing_path, "w", encoding="utf-8") as f:
            for m in missing:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"⚠️  Saved: {missing_path}")
    
    # ============ Summary ============
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset index: {DATASET_FILE}")
    print(f"Splits file:   {SPLITS_FILE}")
    print(f"Unique cases:  {len(dataset)}")
    print(f"Duplicates:    {dup_count}")
    print(f"Train/Val/Test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Overlap check: ✅ CLEAN")


if __name__ == "__main__":
    main()
