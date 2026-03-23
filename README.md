# WSI Report Generation Pipeline

Retrieval-based pathology report generation from Whole Slide Images (WSI) in SVS format using patch-based ResNet50 features, structured field extraction, and semantic reranking.

## Dataset Statistics

| Item | Count |
|------|-------|
| **Total Unique Cases** | **204** |
| Train | 122 (60%) |
| Val | 40 (20%) |
| Test | 42 (20%) |
| Split Overlap | **0** (verified) |

> **Note:** Originally 219 SVS files on disk; 15 duplicates (same TCGA case ID across different UUID folders) were deduplicated. Splits have zero overlap (verified via assertion).

## Data Locations

| Data | Location |
|------|----------|
| SVS Files | `Z:\Kevin\USYD\dataset\{uuid}\*.svs` |
| Reference Reports | `code/reference/TCGA-BRCA/{case_id}/Report.txt` |
| Pipeline Outputs | `code/data/` |

All paths configured in `config.py` (auto-detects WSL vs Windows).

## Pipeline Overview

```
SVS Image → Verify Pairing → Build Index → Build Targets → Batch Pipeline → Baselines → Evaluate
```

| Step | Script | Description |
|------|--------|-------------|
| 0 | `config.py` | **Shared configuration** (paths, settings, progress tracker) |
| 1 | `verify_pair.py` | Verify SVS-Report pairing by filename matching |
| 2 | `build_index.py` | Build dataset index (incremental, **deduplicates** case IDs) |
| 3 | `build_targets.py` | Extract DIAGNOSIS section as training targets |
| 4 | `run_pipeline_all.py` | Batch extract patches/features (**with progress tracking**) |
| 5 | `build_structured_text.py` | Extract structured fields + keywords |
| 6 | `baseline_retrieval.py` | NN retrieval baseline (top-1, raw text) |
| 7 | `baseline_structured.py` | Structured output baseline (top-1, schema template) |
| 8 | `baseline_topk_fusion.py` | **Top-k fusion** (k=3,5) — aggregate fields from multiple cases |
| 9 | `baseline_rerank.py` | **Semantic reranking** — two-stage visual+text retrieval |
| 10 | `evaluate_metrics.py` | ROUGE + keyword coverage + BERTScore evaluation |
| 11 | `compare_baselines.py` | Full comparison table + LaTeX output |
| 12 | `build_modular_labels.py` | **Modular labels** — extract 11 binary clinical concept labels from structured text |
| 13 | `train_modular_classifier.py` | **Stage-A classifier** — MLP(2048→512→11) trained on WSI embeddings → concept labels |

## Progress Tracking

The pipeline tracks completed cases in `data/progress.jsonl`:
- Run `run_pipeline_all.py` tomorrow → only processes NEW cases
- Interrupt and resume anytime without losing progress

## Data Folder Structure

```
code/
├── config.py                        # 🆕 SHARED CONFIG
├── reference/                       # Pathology reports
│   └── TCGA-BRCA/
│       └── TCGA-XX-XXXX/
│           ├── TCGA-XX-XXXX.*.PDF   # Report PDF
│           └── Report.txt           # OCR text
├── data/                            # Pipeline outputs
│   ├── dataset.jsonl                # Dataset index (204 unique cases)
│   ├── splits.json                  # Train/Val/Test splits (0 overlap)
│   ├── progress.jsonl               # 🆕 Progress tracking
│   ├── targets_diagnosis.jsonl      # Training targets
│   ├── features/{case_id}.npy       # [N, 2048] patch features
│   ├── wsi/{case_id}.npy            # [2048] WSI embedding
│   ├── masks/{case_id}.png          # Tissue masks
│   ├── text/{case_id}.json          # Structured fields + keywords
│   └── modular_labels.jsonl         # Multi-label targets (11 clinical concepts)
├── runs/                            # Experiment outputs
│   ├── baseline_retrieval/
│   └── baseline_structured/
└── output/                          # Debug outputs
```

## Quick Start (WSL)

```bash
cd /mnt/c/Users/lawal/Dropbox/Study/USYD/Maphil/code

# Check configuration
python3 config.py

# Build/update index (incremental)
python3 build_index.py

# Process features (incremental, resumes from progress)
python3 run_pipeline_all.py

# Build targets
python3 build_targets.py

# Run baselines
python3 baseline_retrieval.py
python3 baseline_structured.py

# Evaluate
python3 evaluate_metrics.py --exp baseline_structured --split test
```

## Final Output Format

### A. Data Index (for reproducibility)

**data/dataset.jsonl** - One case per line:
```json
{
  "case_id": "TCGA-A7-A4SF",
  "svs_path": "/mnt/z/Kevin/USYD/dataset/<uuid>/TCGA-A7-A4SF-...svs",
  "report_path": "reference/TCGA-BRCA/TCGA-A7-A4SF/Report.txt",
  "num_patches": 50,
  "patch_level": 1,
  "patch_size": 256
}
```

**data/splits.json** - Train/Val/Test split:
```json
{
  "train": ["TCGA-...", ...],   // 131 cases
  "val":   ["TCGA-...", ...],   // 43 cases
  "test":  ["TCGA-...", ...]    // 45 cases
}
```

### B. Visual Features (for training)

- `data/features/{case_id}.npy` - Shape: [N, 2048] (patch features)
- `data/wsi/{case_id}.npy` - Shape: [2048] (mean-pooled WSI feature)

### C. Model Outputs (for evaluation)

**runs/<exp_name>/predictions_test.jsonl**:
```json
{
  "case_id": "TCGA-A7-A4SF",
  "gt": "Report ground truth text ...",
  "pred": "Generated report ..."
}
```

**runs/<exp_name>/metrics.json**:
```json
{
  "ROUGE-L": 0.21,
  "BERTScore-F1": 0.78,
  "CIDEr": 0.12
}
```

### D. Results Table (for paper)

**results/results_table.csv**:
| Method | ROUGE-L | BERTScore-F1 | CIDEr |
|--------|---------|--------------|-------|
| Baseline (WSI→T5) | ... | ... | ... |
| + Structured Prompt | ... | ... | ... |
| + Semantic Loss (Ours) | ... | ... | ... |

## Requirements

```bash
pip install openslide-python pillow numpy torch torchvision tqdm
```

> **Note:** OpenSlide requires system libraries. On Ubuntu/WSL:
> ```bash
> sudo apt-get install openslide-tools
> ```

## Usage

### Step 0: Verify SVS-Report Pairing

```bash
python verify_pair.py
```

Verifies that each SVS file has a matching report by case ID (filename matching).

---

### Step 1: Build Dataset Index

```bash
python build_index.py
```

**What it does:**
- Scans all SVS files in `test_dataset/`
- Matches with reports in `reference/TCGA-BRCA/<case_id>/`
- Creates train/val/test splits (80/10/10)

**Output:**
- `data/dataset.jsonl` - Dataset index
- `data/splits.json` - Train/Val/Test splits
- `data/missing.txt` - Cases with missing data

---

### Step 2: Build Training Targets

```bash
python build_targets.py
```

**What it does:**
- Extracts DIAGNOSIS section from each Report.txt
- Creates clean training targets (shorter, more structured)
- Falls back to first 40 lines if DIAGNOSIS not found

**Why DIAGNOSIS only:**
- Most clean and structured section
- Shorter text → easier to train
- More stable metrics
- Better for presentation

**Output:**
- `data/targets_diagnosis.jsonl` - Training targets

**Format:**
```json
{"case_id": "TCGA-A7-A4SF", "target": "Breast, right, mastectomy: Infiltrating ductal carcinoma..."}
```

---

### Step 3: Batch Feature Extraction

```bash
python run_pipeline_all.py
```

**What it does:**
- Processes all cases in `dataset.jsonl`
- For each case: patches → features → WSI feature
- Skips already processed cases

**Output:**
- `data/features/{case_id}.npy` - Patch features [N, 2048]
- `data/wsi/{case_id}.npy` - WSI feature [2048]
- `data/masks/{case_id}.png` - Tissue mask

---

### Step 4: Retrieval Baseline

```bash
python baseline_retrieval.py
```

**What it does:**
- For each test case, find most similar training case (by WSI embedding)
- Use that case's diagnosis as prediction
- This is a strong baseline for small datasets

**Output:**
- `runs/baseline_retrieval/predictions_test.jsonl`
- `runs/baseline_retrieval/predictions_val.jsonl`

---

### Step 5: Evaluate Metrics

```bash
python evaluate_metrics.py --exp baseline_retrieval --split test
```

**What it does:**
- Computes ROUGE-1, ROUGE-2, ROUGE-L scores
- Shows per-sample results

**Output:**
- `runs/baseline_retrieval/metrics_test.json`

---

### Step 6: Train Generative Baseline (TODO)

```bash
python train_baseline.py
```

**Baseline approach:**
- Input: `data/wsi/{case_id}.npy` (2048-d vector)
- Model: T5-small
- Method: Linear projection → visual tokens → T5 encoder

---

## Quick Start

```bash
cd code

# 1. Build index
python build_index.py

# 2. Build training targets
python build_targets.py

# 3. Extract features (takes time, uses GPU if available)
python run_pipeline_all.py

# 4. Run retrieval baseline
python baseline_retrieval.py

# 5. Evaluate
python evaluate_metrics.py --exp baseline_retrieval --split test
```

## Current Results (All Baselines — Clean, No Leakage)

### Test Set (42 queries, 122 train cases)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | KW Coverage |
|--------|---------|---------|---------|-------------|
| Raw Retrieval | 0.1490 | 0.0393 | 0.0912 | — |
| Structured (top-1) | 0.2582 | 0.1005 | **0.1977** | 0.4544 |
| Structured Fusion (k=3) | 0.2507 | 0.0586 | 0.1709 | 0.5129 |
| Structured Fusion (k=5) | 0.2599 | 0.0617 | 0.1746 | 0.5293 |
| Semantic Rerank (k=1) | 0.2560 | 0.0702 | 0.1726 | 0.5221 |
| Semantic Rerank+Fusion (k=3) | 0.2551 | 0.0632 | 0.1691 | **0.5389** |

### Validation Set (40 queries, 122 train cases)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | KW Coverage |
|--------|---------|---------|---------|-------------|
| Raw Retrieval | 0.0771 | 0.0146 | 0.0500 | — |
| Structured (top-1) | 0.2328 | 0.0750 | 0.1678 | 0.4319 |
| Structured Fusion (k=3) | 0.2481 | 0.0617 | 0.1600 | 0.4830 |
| Structured Fusion (k=5) | 0.2508 | 0.0603 | 0.1588 | 0.4889 |
| Semantic Rerank (k=1) | 0.2761 | 0.0825 | **0.1894** | 0.5421 |
| Semantic Rerank+Fusion (k=3) | 0.2783 | 0.0788 | 0.1843 | **0.5789** |

### Key Findings

- **Structured output** improves ROUGE-L by **+117%** over raw retrieval (0.0912 → 0.1977)
- **Top-k fusion** boosts keyword coverage from 45% → 53% by aggregating multiple cases
- **Semantic reranking** achieves best val ROUGE-L (0.1894) and highest KW coverage (57.9%)
- Text-level matching captures clinical terms that pure visual similarity misses
- Val/test results are consistent — no signs of data leakage

## Notes

- **GPU Acceleration:** Feature extraction uses CUDA if available
- **Skip Existing:** `run_pipeline_all.py` skips already processed cases
- **Reproducibility:** Random seed (42), deduplicated case IDs, zero split overlap
- **OCR Noise:** Report.txt contains OCR errors; DIAGNOSIS extraction helps filter noise
- **Sentence Model:** Semantic reranking uses `all-MiniLM-L6-v2` (90 MB, no fine-tuning needed)

## Requirements

```bash
pip install openslide-python pillow numpy torch torchvision tqdm
pip install rouge-score bert-score sentence-transformers
```

## TODO

- [x] Batch processing multiple SVS files
- [x] Extract DIAGNOSIS targets for training
- [x] Retrieval baseline (raw text)
- [x] Structured output baseline (schema template)
- [x] Top-k structured fusion (k=3, k=5)
- [x] Semantic reranking (two-stage visual + text)
- [x] ROUGE + BERTScore + keyword coverage evaluation
- [x] Fix data leakage (dedup case IDs across UUID folders)
- [x] Modular multi-label targets (11 clinical concepts from structured text)
- [x] Stage-A modular classifier (WSI → concept labels, MLP + BCEWithLogitsLoss)
- [ ] Implement train_baseline.py (T5-small + visual tokens)
- [ ] Attention-based MIL aggregation (replace mean-pooling)
- [ ] CLAM/HIPT feature extractors
