"""
Train a multi-label classifier: WSI embedding → 11 diagnostic concept labels.

Simplified Stage-A reproduction of:
  "Enhancing Structured Pathology Report Generation With Foundation Model
   and Modular Design"

Pipeline:
  WSI embedding [2048] → MLP(2048→512→11) → sigmoid → binary concept labels

Usage:
  python3 train_modular_classifier.py
  python3 train_modular_classifier.py --epochs 50 --lr 5e-4 --patience 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Try importing shared config; fall back to hardcoded paths
# ---------------------------------------------------------------------------
try:
    from config import OUTPUT_DIR, WSI_DIR, SPLITS_FILE
except ImportError:
    OUTPUT_DIR = Path(__file__).parent / "data"
    WSI_DIR = OUTPUT_DIR / "wsi"
    SPLITS_FILE = OUTPUT_DIR / "splits.json"

RUNS_DIR = OUTPUT_DIR.parent / "runs"

# ---------------------------------------------------------------------------
# Label schema — fixed order, must match build_modular_labels.py
# ---------------------------------------------------------------------------
LABEL_NAMES: List[str] = [
    "carcinoma_present",
    "benign_present",
    "invasive_present",
    "in_situ_present",
    "lymph_node_positive",
    "grade_1",
    "grade_2",
    "grade_3",
    "ductal_present",
    "lobular_present",
    "metastasis_present",
]
NUM_LABELS = len(LABEL_NAMES)  # 11


# ============================================================================
# Dataset
# ============================================================================

class WSILabelDataset(Dataset):
    """Loads (WSI embedding, label vector) pairs for a given split."""

    def __init__(
        self,
        case_ids: List[str],
        labels_by_case: Dict[str, Dict[str, int]],
        wsi_dir: Path,
    ) -> None:
        self.samples: List[Tuple[str, np.ndarray, np.ndarray]] = []
        skipped = 0
        for cid in case_ids:
            emb_path = wsi_dir / f"{cid}.npy"
            if cid not in labels_by_case:
                skipped += 1
                continue
            if not emb_path.exists():
                skipped += 1
                continue
            emb = np.load(emb_path).astype(np.float32)          # [2048]
            lab = np.array(
                [labels_by_case[cid].get(k, 0) for k in LABEL_NAMES],
                dtype=np.float32,
            )                                                     # [11]
            self.samples.append((cid, emb, lab))
        if skipped:
            print(f"  ⚠ Skipped {skipped} case(s) (missing embedding or label)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        cid, emb, lab = self.samples[idx]
        return cid, torch.from_numpy(emb), torch.from_numpy(lab)


def collate_fn(batch):
    """Custom collate to keep case_id strings alongside tensors."""
    cids, embs, labs = zip(*batch)
    return list(cids), torch.stack(embs), torch.stack(labs)


# ============================================================================
# Model
# ============================================================================

class ConceptClassifier(nn.Module):
    """Simple MLP: 2048 → 512 → 11 (logits)."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512,
                 output_dim: int = NUM_LABELS, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, 11] logits


# ============================================================================
# Metrics (no sklearn dependency)
# ============================================================================

def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_metrics(
    gt: np.ndarray, pred: np.ndarray, probs: np.ndarray, loss: float
) -> Dict:
    """Compute per-label and aggregate metrics.

    Args:
        gt:    [N, 11] binary ground truth
        pred:  [N, 11] binary predictions
        probs: [N, 11] sigmoid probabilities
        loss:  scalar average loss
    """
    n, k = gt.shape

    # Per-label
    per_label_acc: Dict[str, float] = {}
    per_label_f1: Dict[str, float] = {}
    per_label_precision: Dict[str, float] = {}
    per_label_recall: Dict[str, float] = {}

    for i, name in enumerate(LABEL_NAMES):
        tp = float(((gt[:, i] == 1) & (pred[:, i] == 1)).sum())
        fp = float(((gt[:, i] == 0) & (pred[:, i] == 1)).sum())
        fn = float(((gt[:, i] == 1) & (pred[:, i] == 0)).sum())
        tn = float(((gt[:, i] == 0) & (pred[:, i] == 0)).sum())

        acc = _safe_div(tp + tn, tp + fp + fn + tn)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)

        per_label_acc[name] = round(acc, 4)
        per_label_f1[name] = round(f1, 4)
        per_label_precision[name] = round(prec, 4)
        per_label_recall[name] = round(rec, 4)

    # Macro F1 = mean of per-label F1
    macro_f1 = float(np.mean(list(per_label_f1.values())))

    # Micro F1 = global TP / FP / FN across all labels
    tp_all = float(((gt == 1) & (pred == 1)).sum())
    fp_all = float(((gt == 0) & (pred == 1)).sum())
    fn_all = float(((gt == 1) & (pred == 0)).sum())
    micro_prec = _safe_div(tp_all, tp_all + fp_all)
    micro_rec = _safe_div(tp_all, tp_all + fn_all)
    micro_f1 = _safe_div(2 * micro_prec * micro_rec, micro_prec + micro_rec)

    return {
        "loss": round(loss, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_label_accuracy": per_label_acc,
        "per_label_f1": per_label_f1,
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
        "num_cases": n,
    }


# ============================================================================
# Evaluation loop
# ============================================================================

@torch.no_grad()
def evaluate(
    model: ConceptClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict, List[Dict]]:
    """Run evaluation, return (metrics_dict, per_case_predictions)."""
    model.eval()
    all_cids: List[str] = []
    all_gt: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for cids, embs, labels in loader:
        embs = embs.to(device)
        labels = labels.to(device)
        logits = model(embs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_cids.extend(cids)
        all_gt.append(labels.cpu().numpy())
        all_probs.append(probs)

    gt = np.concatenate(all_gt, axis=0)        # [N, 11]
    probs = np.concatenate(all_probs, axis=0)  # [N, 11]
    pred = (probs >= 0.5).astype(np.float32)   # [N, 11]
    avg_loss = total_loss / max(n_batches, 1)

    metrics = compute_metrics(gt, pred, probs, avg_loss)

    # Per-case predictions
    predictions: List[Dict] = []
    for i, cid in enumerate(all_cids):
        predictions.append({
            "case_id": cid,
            "labels_gt": {LABEL_NAMES[j]: int(gt[i, j]) for j in range(NUM_LABELS)},
            "labels_pred": {LABEL_NAMES[j]: int(pred[i, j]) for j in range(NUM_LABELS)},
            "probs": {LABEL_NAMES[j]: round(float(probs[i, j]), 4) for j in range(NUM_LABELS)},
        })

    return metrics, predictions


# ============================================================================
# Class imbalance weighting
# ============================================================================

MAX_POS_WEIGHT = 10.0  # Clamp to avoid loss explosion on very rare labels


def compute_pos_weight(dataset: WSILabelDataset, device: torch.device) -> torch.Tensor:
    """Compute per-label pos_weight = num_neg / num_pos for BCEWithLogitsLoss.

    Labels with zero positives get pos_weight = MAX_POS_WEIGHT.
    All weights are clamped to [1.0, MAX_POS_WEIGHT].
    """
    n = len(dataset)
    if n == 0:
        return torch.ones(NUM_LABELS, device=device)

    # Stack all label vectors  → [N, 11]
    all_labels = np.array([dataset.samples[i][2] for i in range(n)])
    pos_counts = all_labels.sum(axis=0)  # [11]

    weights = []
    for i, name in enumerate(LABEL_NAMES):
        pos = pos_counts[i]
        neg = n - pos
        if pos == 0:
            w = MAX_POS_WEIGHT
        else:
            w = float(neg / pos)
        w = max(1.0, min(w, MAX_POS_WEIGHT))
        weights.append(w)

    pw = torch.tensor(weights, dtype=torch.float32, device=device)

    # Print summary
    print(f"\n  Class weights (pos_weight = num_neg / num_pos, "
          f"clamped to [{1.0}, {MAX_POS_WEIGHT}]):")
    for i, name in enumerate(LABEL_NAMES):
        pos = int(pos_counts[i])
        print(f"    {name:<25s}  pos={pos:>3d}/{n}  weight={pw[i]:.2f}")
    print()

    return pw


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(
    model: ConceptClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for _, embs, labels in loader:
        embs = embs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(embs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Stage-A modular classifier: WSI → concept labels.",
    )
    p.add_argument("--labels_path", type=Path,
                    default=OUTPUT_DIR / "modular_labels.jsonl")
    p.add_argument("--wsi_dir", type=Path, default=WSI_DIR)
    p.add_argument("--splits_path", type=Path, default=SPLITS_FILE)
    p.add_argument("--run_dir", type=Path,
                    default=RUNS_DIR / "modular_classifier")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()

    # ── Seed ──────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load labels ───────────────────────────────────────────────────────
    labels_by_case: Dict[str, Dict[str, int]] = {}
    with open(args.labels_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            labels_by_case[rec["case_id"]] = rec["labels"]
    print(f"Loaded labels for {len(labels_by_case)} cases")

    # ── Load splits ───────────────────────────────────────────────────────
    with open(args.splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]
    print(f"Splits — train: {len(train_ids)}, val: {len(val_ids)}, "
          f"test: {len(test_ids)}")

    # ── Build datasets ────────────────────────────────────────────────────
    print("Loading train set...")
    train_ds = WSILabelDataset(train_ids, labels_by_case, args.wsi_dir)
    print("Loading val set...")
    val_ds = WSILabelDataset(val_ids, labels_by_case, args.wsi_dir)
    print("Loading test set...")
    test_ds = WSILabelDataset(test_ids, labels_by_case, args.wsi_dir)

    if len(train_ds) == 0:
        print("ERROR: No training samples found.", file=sys.stderr)
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    # ── Model / loss / optimizer ──────────────────────────────────────────
    model = ConceptClassifier().to(device)

    # Class-weighted BCE to handle label imbalance
    pos_weight = compute_pos_weight(train_ds, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # ── Output directory ──────────────────────────────────────────────────
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_f1 = -1.0
    patience_counter = 0
    train_log: List[Dict] = []

    print(f"\n{'='*60}")
    print(f"  Training — {args.epochs} epochs, patience={args.patience}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        val_metrics, _ = evaluate(model, val_loader, criterion, device)

        is_best = val_metrics["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            patience_counter += 1

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "is_best": is_best,
        }
        train_log.append(log_entry)

        star = " ★" if is_best else ""
        print(f"  Epoch {epoch:>2d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
              f"val_micro_f1={val_metrics['micro_f1']:.4f}{star}")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # Save training log
    with open(run_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

    # ── Final evaluation with best checkpoint ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Evaluating best checkpoint (val macro_f1={best_val_f1:.4f})")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(run_dir / "best.pt",
                                     map_location=device, weights_only=True))

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        metrics, predictions = evaluate(model, loader, criterion, device)

        # Save metrics
        metrics_path = run_dir / f"metrics_{split_name}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Save predictions
        pred_path = run_dir / f"predictions_{split_name}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"\n  {split_name.upper()} — {metrics['num_cases']} cases")
        print(f"    loss={metrics['loss']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  "
              f"micro_f1={metrics['micro_f1']:.4f}")
        print(f"    Per-label F1:")
        for name in LABEL_NAMES:
            f1 = metrics["per_label_f1"][name]
            acc = metrics["per_label_accuracy"][name]
            print(f"      {name:<25s}  F1={f1:.4f}  Acc={acc:.4f}")

    print(f"\n  Outputs saved to: {run_dir}")
    print()


if __name__ == "__main__":
    main()
