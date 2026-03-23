"""
Verify SVS-Report pairing by filename matching (hard alignment).
- SVS files are in: test_dataset/<uuid>/*.svs
- Reports are in: reference/TCGA-BRCA/<case_id>/*.PDF

Matching rule: Extract case ID (e.g., TCGA-A7-A4SF) from both filenames.
No OCR needed - just filename comparison.
"""

import re
from pathlib import Path

# ============ Configuration ============
SVS_DIR = Path(r"test_dataset")
REF_DIR = Path(r"reference/TCGA-BRCA")  # All TCGA-BRCA cases are here


def get_case_id_from_name(name: str):
    """Extract TCGA case ID (e.g., TCGA-A7-A4SF) from filename."""
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", name)
    return m.group(1) if m else None


# ============ Find all SVS files ============
svs_files = list(SVS_DIR.rglob("*.svs"))

if not svs_files:
    print(f"No SVS files found in {SVS_DIR}")
    print("Add SVS files to test_dataset/ folder first.")
    exit(1)

print(f"Found {len(svs_files)} SVS file(s) in {SVS_DIR}")
print("=" * 70)

# ============ Verify each SVS file ============
verified_count = 0
failed_count = 0

for svs_path in svs_files:
    svs_name = svs_path.name
    svs_case = get_case_id_from_name(svs_name)

    print(f"\nSVS File:  {svs_name}")

    if not svs_case:
        print(f"❌ FAILED: Cannot parse case ID from filename")
        failed_count += 1
        continue

    print(f"Case ID:   {svs_case}")

    # Look for matching reference folder
    ref_case_dir = REF_DIR / svs_case

    if not ref_case_dir.exists():
        print(f"❌ FAILED: Reference folder not found")
        print(f"           Expected: {ref_case_dir}")
        failed_count += 1
        continue

    # Find PDF files in reference folder
    pdfs = list(ref_case_dir.glob("*.pdf")) + list(ref_case_dir.glob("*.PDF"))

    if not pdfs:
        print(f"❌ FAILED: No PDF found in {ref_case_dir}")
        failed_count += 1
        continue

    # Extract case ID from PDF filenames
    pdf_matches = []
    for pdf in pdfs:
        pdf_case = get_case_id_from_name(pdf.name)
        if pdf_case:
            pdf_matches.append((pdf.name, pdf_case))

    print(f"PDF(s):    {[p.name for p in pdfs]}")

    # Check if any PDF filename contains the same case ID
    matched_pdfs = [p for p, c in pdf_matches if c == svs_case]

    if matched_pdfs:
        print(f"✅ VERIFIED: SVS case ID matches PDF filename")
        print(f"             Matched PDF: {matched_pdfs[0]}")
        verified_count += 1
    else:
        print(f"⚠️  WARNING: PDF filename does not contain case ID")
        print(f"             May need manifest/metadata mapping to verify")
        failed_count += 1

# ============ Summary ============
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total SVS files:  {len(svs_files)}")
print(f"Verified:         {verified_count}")
print(f"Failed/Warning:   {failed_count}")

if verified_count == len(svs_files):
    print("\n✅ All SVS files verified successfully!")
else:
    print(f"\n⚠️  {failed_count} file(s) need attention.")