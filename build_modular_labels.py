"""
Build structured multi-label targets for modular pathology report generation.

Converts extracted structured text fields (from data/text/{case_id}.json)
into binary clinical concept labels (11 labels), following the modular design
approach from:
  "Enhancing Structured Pathology Report Generation With Foundation Model
   and Modular Design"

Label schema (11 binary concepts):
  carcinoma_present, benign_present, invasive_present, in_situ_present,
  lymph_node_positive, grade_1, grade_2, grade_3,
  ductal_present, lobular_present, metastasis_present

NOTE: margin_positive was dropped — TCGA-BRCA reports almost universally say
"margins negative / uninvolved", yielding 0% positive rate.  An all-zero
label would collapse classifier loss during training.

Stage A of a two-stage pipeline:
  Stage A = WSI → structured labels  (this script)
  Stage B = labels → T5 report generation  (downstream)

Usage:
  python3 build_modular_labels.py
  python3 build_modular_labels.py --text_dir data/text --out data/modular_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Try importing from config.py for default paths; fall back to hardcoded
# ---------------------------------------------------------------------------
try:
    from config import OUTPUT_DIR, TEXT_DIR
except ImportError:
    OUTPUT_DIR = Path(__file__).parent / "data"
    TEXT_DIR = OUTPUT_DIR / "text"


# ============================================================================
# Text normalisation
# ============================================================================

def normalise(text: Optional[str]) -> str:
    """Lowercase, strip newlines, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================================
# Individual label extraction helpers
# ============================================================================

def _has_any_word(text: str, patterns: List[str]) -> bool:
    """Return True if *text* contains any *pattern* as a whole-word match.
    Each pattern is wrapped with \\b word boundaries to avoid partial hits
    like 'carcinoma-like' or substring false positives from OCR noise."""
    return any(re.search(r"\b" + re.escape(p) + r"\b", text) for p in patterns)


def detect_carcinoma(full_text: str) -> int:
    """A. carcinoma_present"""
    return int(_has_any_word(full_text, [
        "carcinoma", "adenocarcinoma",
        "ductal carcinoma", "lobular carcinoma",
    ]))


def detect_benign(full_text: str) -> int:
    """B. benign_present (can co-occur with carcinoma)."""
    return int(_has_any_word(full_text, [
        "benign", "fibroadenoma", "hyperplasia",
    ]))


def detect_invasive(full_text: str) -> int:
    """C. invasive_present"""
    return int(_has_any_word(full_text, ["invasive", "infiltrating"]))


def detect_in_situ(full_text: str) -> int:
    """D. in_situ_present"""
    return int(_has_any_word(full_text, ["in situ", "dcis", "lcis"]))


def detect_ductal(full_text: str) -> int:
    """E. ductal_present"""
    return int(bool(re.search(r"\bductal\b", full_text)))


def detect_lobular(full_text: str) -> int:
    """F. lobular_present"""
    return int(bool(re.search(r"\blobular\b", full_text)))


def detect_metastasis(full_text: str) -> int:
    """G. metastasis_present — positive when metastasis/metastatic appears
    AND there is no *global* negation.  Local negations scoped to lymph nodes
    (e.g. 'no metastasis in lymph nodes') are NOT treated as global negation,
    so 'no metastasis in lymph nodes but distant metastasis present' → 1."""

    # Local negations (scoped — do NOT kill the whole label)
    local_neg_patterns = [
        "no metastasis in lymph",
        "no lymph node metastasis",
        "negative for metastatic carcinoma in lymph",
    ]

    # Global negations (unscoped — kill the label)
    global_neg_patterns = [
        "without metastasis",
        "negative for metastasis",
        "no metastatic carcinoma",
        "no evidence of metastasis",
        "no metastatic disease",
    ]

    pos_patterns = ["metastasis", "metastatic"]

    has_pos = _has_any_word(full_text, pos_patterns)
    if not has_pos:
        return 0

    # Check global negation — but first strip local negation phrases so they
    # don't trigger "no metastasis" as a global neg
    text_for_neg = full_text
    for lp in local_neg_patterns:
        text_for_neg = text_for_neg.replace(lp, "")

    has_global_neg = _has_any_word(text_for_neg, global_neg_patterns)
    # Also catch bare "no metastasis" only in the cleaned text
    has_global_neg = has_global_neg or bool(
        re.search(r"\bno\s+metastasis\b", text_for_neg)
    )

    if has_pos and not has_global_neg:
        return 1
    return 0


def detect_lymph_node_positive(lymph_nodes_text: str, full_text: str) -> int:
    """H. lymph_node_positive — use LYMPH_NODES field ONLY.
    If no dedicated LYMPH_NODES field exists, return 0 rather than
    falling back to full_text (which risks false positives from
    unrelated phrases like 'margins positive').
    Negative patterns (especially 0/N) dominate when both signals present."""
    if not lymph_nodes_text:
        return 0

    text = lymph_nodes_text

    neg_patterns = [
        "negative", "no metastasis",
        "no lymph node metastasis", "negative for metastatic",
    ]
    # 0/N pattern  e.g.  0/3, 0/10
    has_zero_ratio = bool(re.search(r"\b0\s*/\s*\d+", text))
    has_neg = _has_any_word(text, neg_patterns) or has_zero_ratio

    pos_patterns = ["positive", "metastatic", "metastasis", "involved"]
    # N/M with N >= 1  e.g.  1/3, 2/10
    has_pos_ratio = bool(re.search(r"\b[1-9]\d*\s*/\s*\d+", text))
    has_pos = _has_any_word(text, pos_patterns) or has_pos_ratio

    # 0/x dominates — if both signals present, negative wins for 0/x cases
    if has_zero_ratio:
        return 0
    if has_neg and has_pos:
        return 0
    return int(has_pos)


# NOTE: margin_positive dropped — TCGA-BRCA reports almost universally say
# "margins negative / uninvolved", yielding 0% positive rate across 204 cases.
# An all-zero label collapses any classifier loss.  Kept out of the schema
# intentionally; can be re-added if a dataset with margin variation is used.


def extract_grade(full_text: str) -> int:
    """J. Extract histological grade (0 = unknown, 1/2/3).
    Priority: grade 3 > grade 2 > grade 1.
    Uses strict word-boundary regex so 'grade i' does NOT match 'grade ii'."""
    g3 = bool(re.search(r"\bgrade\s*3\b", full_text)) or \
         bool(re.search(r"\bgrade\s*iii\b", full_text)) or \
         _has_any_word(full_text, ["poorly differentiated"])
    g2 = bool(re.search(r"\bgrade\s*2\b", full_text)) or \
         bool(re.search(r"\bgrade\s*ii\b", full_text)) or \
         _has_any_word(full_text, ["moderately differentiated"])
    g1 = bool(re.search(r"\bgrade\s*1\b", full_text)) or \
         bool(re.search(r"\bgrade\s*i\b", full_text)) or \
         _has_any_word(full_text, ["well differentiated"])

    if g3:
        return 3
    if g2:
        return 2
    if g1:
        return 1
    return 0


# ============================================================================
# Main label builder
# ============================================================================

def build_labels(rec: Dict) -> Dict[str, int]:
    """Given a loaded text/{case_id}.json record, return a label dict."""
    # Normalise individual fields
    diagnosis = normalise(rec.get("diagnosis", ""))
    microscopic = normalise(rec.get("microscopic", ""))
    margins = normalise(rec.get("margins", ""))
    lymph_nodes = normalise(rec.get("lymph_nodes", ""))
    stage = normalise(rec.get("ptnm", ""))  # field is named 'ptnm' in json

    full_text = " ".join([diagnosis, microscopic, margins, lymph_nodes, stage])

    grade = extract_grade(full_text)

    labels: Dict[str, int] = {
        "carcinoma_present":    detect_carcinoma(full_text),
        "benign_present":       detect_benign(full_text),
        "invasive_present":     detect_invasive(full_text),
        "in_situ_present":      detect_in_situ(full_text),
        "lymph_node_positive":  detect_lymph_node_positive(lymph_nodes, full_text),
        "grade_1":              int(grade == 1),
        "grade_2":              int(grade == 2),
        "grade_3":              int(grade == 3),
        "ductal_present":       detect_ductal(full_text),
        "lobular_present":      detect_lobular(full_text),
        "metastasis_present":   detect_metastasis(full_text),
    }
    return labels


def build_source_text(rec: Dict) -> Dict[str, str]:
    """Return the raw (unnormalised) source fields for provenance."""
    return {
        "DIAGNOSIS":    rec.get("diagnosis", ""),
        "MICROSCOPIC":  rec.get("microscopic", ""),
        "MARGINS":      rec.get("margins", ""),
        "LYMPH_NODES":  rec.get("lymph_nodes", ""),
        "STAGE":        rec.get("ptnm", ""),
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build modular multi-label targets from structured text fields.",
    )
    p.add_argument(
        "--text_dir",
        type=Path,
        default=TEXT_DIR,
        help="Directory containing text/{case_id}.json files (default: config.TEXT_DIR)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / "modular_labels.jsonl",
        help="Output JSONL path (default: data/modular_labels.jsonl)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text_dir: Path = args.text_dir
    out_path: Path = args.out

    # Collect input files
    json_files = sorted(text_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: No .json files found in {text_dir}", file=sys.stderr)
        sys.exit(1)

    # Counters for summary
    label_names = [
        "carcinoma_present", "benign_present", "invasive_present",
        "in_situ_present", "lymph_node_positive",
        "grade_1", "grade_2", "grade_3",
        "ductal_present", "lobular_present", "metastasis_present",
    ]
    positive_counts: Dict[str, int] = {k: 0 for k in label_names}

    # Process
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                rec = json.load(f)

            case_id = rec.get("case_id", jf.stem)
            labels = build_labels(rec)
            source = build_source_text(rec)

            row = {
                "case_id": case_id,
                "labels": labels,
                "source_text": source,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            for k in label_names:
                positive_counts[k] += labels[k]

    # Summary
    n = len(json_files)
    print(f"\n{'='*55}")
    print(f"  Modular labels built — {n} cases → {out_path.name}")
    print(f"{'='*55}")
    print(f"  {'Label':<25s} {'Positive':>8s} {'Rate':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    for k in label_names:
        cnt = positive_counts[k]
        rate = cnt / n if n else 0
        print(f"  {k:<25s} {cnt:>8d} {rate:>7.1%}")
    print(f"{'='*55}")
    print(f"  Output: {out_path}")
    print()


if __name__ == "__main__":
    main()
