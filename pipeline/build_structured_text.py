import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Extract structured fields from pathology reports.
Outputs: data/text/{case_id}.json

Fields extracted:
- diagnosis: DIAGNOSIS section
- microscopic: MICROSCOPIC DESCRIPTION section  
- margins: MARGINS information
- lymph_nodes: LYMPH NODE findings
- ptnm: pathologic TNM staging
- keywords: key medical terms (for semantic coverage)
- full: full text (fallback)
"""

import json
import re
from pathlib import Path
from collections import Counter

# Import shared configuration
from config import DATASET_FILE, TEXT_DIR, print_config


def clean_text(text: str) -> str:
    """Clean OCR noise and normalize whitespace."""
    # Remove common OCR artifacts
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)  # Keep only printable ASCII
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


def extract_section(text: str, section_names: list) -> str:
    """
    Extract a section from the report.
    section_names: list of possible headers (e.g., ['DIAGNOSIS', 'DIAGNOSES'])
    """
    # Build pattern for section header
    headers = '|'.join(re.escape(h) for h in section_names)
    
    # Pattern: section header followed by colon, then content until next header or end
    pattern = rf'(?:{headers})\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z\s\(\)\/\-]{{3,}}[:\-]|\Z)'
    
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return clean_text(match.group(1))
    return ""


def extract_keywords(text: str, top_k: int = 20) -> list:
    """
    Extract key medical terms using simple noun phrase extraction.
    Returns top-K most frequent meaningful terms.
    """
    # Common medical terms to look for
    medical_patterns = [
        r'\b(carcinoma|adenocarcinoma|ductal|lobular|invasive|infiltrating)\b',
        r'\b(grade\s*[I1-3]+|nottingham\s*\d+)\b',
        r'\b(positive|negative)\b',
        r'\b(metastatic|metastasis)\b',
        r'\b(margin[s]?|resection)\b',
        r'\b(lymph\s*node[s]?)\b',
        r'\b(ER|PR|HER2|HER-2|Ki-67)\b',
        r'\b(pT\d|pN\d|pM\d|stage\s*[IV]+)\b',
        r'\b(breast|axillary|sentinel)\b',
        r'\b(mastectomy|lumpectomy|excision|biopsy)\b',
        r'\b(tumor|neoplasm|mass)\b',
        r'\b(\d+\.?\d*\s*[cx]m)\b',  # measurements
    ]
    
    keywords = []
    text_lower = text.lower()
    
    for pattern in medical_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        keywords.extend(matches)
    
    # Count and return top-K
    counter = Counter(keywords)
    return [kw for kw, count in counter.most_common(top_k)]


def extract_structured_fields(report_text: str) -> dict:
    """Extract all structured fields from a report."""
    
    # Normalize text
    text = report_text.replace('\r', '\n')
    
    # Extract sections
    diagnosis = extract_section(text, [
        'DIAGNOSIS', 'DIAGNOSES', 'DIAGNOSIS(ES)', 'FINAL DIAGNOSIS'
    ])
    
    microscopic = extract_section(text, [
        'MICROSCOPIC DESCRIPTION', 'MICROSCOPIC', 'MICROSCOPIC EXAMINATION',
        'MICROSCOPIC FINDINGS'
    ])
    
    margins = extract_section(text, [
        'MARGINS', 'SURGICAL MARGINS', 'MARGIN STATUS'
    ])
    
    lymph_nodes = extract_section(text, [
        'LYMPH NODE', 'LYMPH NODES', 'SENTINEL NODE', 'AXILLARY'
    ])
    
    # Extract pTNM staging
    ptnm_match = re.search(r'p?T\d[a-z]?\s*N\d[a-z]?\s*M[0-9x]', text, re.IGNORECASE)
    ptnm = ptnm_match.group(0) if ptnm_match else ""
    
    # If no sections found, use first part of text as diagnosis
    if not diagnosis:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        diagnosis = clean_text(' '.join(lines[:20]))
    
    # Extract keywords from full text
    keywords = extract_keywords(text)
    
    # Full text (cleaned)
    full = clean_text(text)
    
    return {
        'diagnosis': diagnosis,
        'microscopic': microscopic,
        'margins': margins,
        'lymph_nodes': lymph_nodes,
        'ptnm': ptnm,
        'keywords': keywords,
        'full': full[:2000]  # Limit full text length
    }


def main():
    print_config()
    
    # Use TEXT_DIR from config
    OUTPUT_DIR = TEXT_DIR
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if not DATASET_FILE.exists():
        print(f"❌ Dataset file not found: {DATASET_FILE}")
        return
    
    items = []
    with DATASET_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    
    print(f"Processing {len(items)} cases...")
    print("=" * 60)
    
    success_count = 0
    stats = {
        'has_diagnosis': 0,
        'has_microscopic': 0,
        'has_margins': 0,
        'has_lymph_nodes': 0,
        'has_ptnm': 0
    }
    
    for item in items:
        case_id = item['case_id']
        report_path = Path(item['report_path'])
        
        if not report_path.exists():
            print(f"⚠️  {case_id}: Report not found")
            continue
        
        # Read report
        report_text = report_path.read_text(errors='ignore')
        
        # Extract structured fields
        fields = extract_structured_fields(report_text)
        fields['case_id'] = case_id
        
        # Update stats
        if fields['diagnosis']: stats['has_diagnosis'] += 1
        if fields['microscopic']: stats['has_microscopic'] += 1
        if fields['margins']: stats['has_margins'] += 1
        if fields['lymph_nodes']: stats['has_lymph_nodes'] += 1
        if fields['ptnm']: stats['has_ptnm'] += 1
        
        # Save to file
        output_path = OUTPUT_DIR / f"{case_id}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(fields, f, indent=2, ensure_ascii=False)
        
        success_count += 1
        
        # Preview first few
        if success_count <= 3:
            print(f"\n✅ {case_id}:")
            print(f"   Diagnosis: {fields['diagnosis'][:80]}...")
            print(f"   Keywords: {fields['keywords'][:5]}")
            print(f"   pTNM: {fields['ptnm'] or 'N/A'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cases processed: {success_count}")
    print(f"\nField extraction rates:")
    for field, count in stats.items():
        pct = count / success_count * 100 if success_count > 0 else 0
        print(f"  {field}: {count}/{success_count} ({pct:.1f}%)")
    
    print(f"\n✅ Saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
