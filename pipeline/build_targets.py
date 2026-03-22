import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Build training targets by extracting DIAGNOSIS section from Report.txt.
Generates: data/targets_diagnosis.jsonl

Why DIAGNOSIS only:
- Most clean and structured section
- Shorter, easier to train
- More stable metrics
- Better for presentation
"""

import json
import re
from pathlib import Path

# Import shared configuration
from config import DATASET_FILE, TARGETS_FILE, print_config


def extract_diagnosis(text: str) -> str:
    """
    Extract DIAGNOSIS section from pathology report.
    Falls back to first ~40 lines if DIAGNOSIS not found.
    """
    # Normalize line endings
    t = text.replace("\r", "\n")
    t = re.sub(r"\n{2,}", "\n", t)

    # Common diagnosis headers pattern
    # Matches: "DIAGNOSIS:", "DIAGNOSIS(ES):", "DIAGNOSES:", etc.
    pat = re.compile(
        r"(DIAGNOSIS\(ES\)|DIAGNOSES|DIAGNOSIS)\s*:\s*(.*?)(\n[A-Z][A-Z \(\)\/\-]{3,}:\s|\Z)",
        re.S | re.IGNORECASE
    )
    m = pat.search(t)
    
    if m:
        diag = m.group(2).strip()
    else:
        # Fallback: first ~40 non-empty lines as crude summary
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        diag = " ".join(lines[:40])

    # Cleanup whitespace
    diag = re.sub(r"\s+", " ", diag).strip()
    return diag


def main():
    print_config()
    
    # Use config paths
    DATASET = DATASET_FILE
    OUT = TARGETS_FILE
    
    # Check dataset exists
    if not DATASET.exists():
        print(f"❌ Dataset file not found: {DATASET}")
        print("   Run build_index.py first.")
        return

    # Load dataset
    items = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    print(f"Loaded {len(items)} cases from {DATASET}")
    print("=" * 60)

    # Extract diagnosis targets
    success_count = 0
    skip_count = 0
    
    with OUT.open("w", encoding="utf-8") as f:
        for it in items:
            case_id = it["case_id"]
            report_path = Path(it["report_path"])
            
            if not report_path.exists():
                print(f"⚠️  {case_id}: Report not found")
                skip_count += 1
                continue
            
            # Read report
            rpt = report_path.read_text(errors="ignore")
            
            # Extract diagnosis
            target = extract_diagnosis(rpt)
            
            # Skip if too short (likely extraction failed)
            if len(target) < 10:
                print(f"⚠️  {case_id}: Target too short ({len(target)} chars)")
                skip_count += 1
                continue
            
            # Write to output
            f.write(json.dumps({
                "case_id": case_id,
                "target": target
            }, ensure_ascii=False) + "\n")
            
            success_count += 1
            
            # Preview first few
            if success_count <= 3:
                preview = target[:100] + "..." if len(target) > 100 else target
                print(f"✅ {case_id}: {preview}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cases:     {len(items)}")
    print(f"Targets saved:   {success_count}")
    print(f"Skipped:         {skip_count}")
    print(f"\n✅ Saved: {OUT}")


if __name__ == "__main__":
    main()
