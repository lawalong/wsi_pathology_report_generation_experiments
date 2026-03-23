"""
Prompt-Guided Generation with Semantic Alignment Loss

Model: T5-small
Input: visual tokens (projected WSI embedding) + text prompt
Output: structured pathology report
Loss: L_total = L_ce + lambda_sem * L_sem

Visual tokens:
  2048-d WSI vector → Linear → (8 × 512) → prepend to encoder inputs

Semantic alignment loss:
  Frozen all-MiniLM-L6-v2 encodes target text → 384-d
  Decoder hidden states → mean pool → project to 384-d
  L_sem = 1 - cosine(pred_emb, target_emb)
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Hyperparameters (locked) ────────────────────────────────────
MODEL_NAME = "t5-small"        # d_model = 512
EPOCHS = 15
BATCH_SIZE = 8
LR = 2e-4
LAMBDA_SEM = 0.2
NUM_VISUAL_TOKENS = 8
D_WSI = 2048
MAX_INPUT_LEN = 384
MAX_TARGET_LEN = 192
PATIENCE = 4                   # early stopping on val ROUGE-L

SAVE_DIR = Path("runs/ours_prompt_semantic")

from config import OUTPUT_DIR

# ── Dataset ─────────────────────────────────────────────────────
class PromptDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_input_len, max_target_len):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load WSI embedding
        wsi_path = Path(s["wsi_path"])
        if not wsi_path.is_absolute():
            # Relative to code/ dir — resolve from config
            wsi_path = OUTPUT_DIR.parent / wsi_path
        wsi_vec = np.load(wsi_path).astype(np.float32)  # (2048,)

        # Tokenize prompt
        enc = self.tokenizer(
            s["prompt_text"],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target (T5 uses same tokenizer for source and target)
        tgt = self.tokenizer(
            text_target=s["target_text"],
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = tgt.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": labels,
            "wsi_vec": torch.from_numpy(wsi_vec),
            "target_text": s["target_text"],
            "case_id": s["case_id"],
        }


# ── Model ───────────────────────────────────────────────────────
class PromptSemanticModel(nn.Module):
    """T5-small with prepended visual tokens and semantic alignment head."""

    def __init__(self, model_name=MODEL_NAME, num_vis_tokens=NUM_VISUAL_TOKENS):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        d_model = self.t5.config.d_model  # 512

        # Visual projection: 2048 → (num_vis_tokens × d_model)
        self.vis_proj = nn.Linear(D_WSI, num_vis_tokens * d_model)
        self.num_vis_tokens = num_vis_tokens
        self.d_model = d_model

        # Semantic alignment projection: d_model → 384 (MiniLM dim)
        self.sem_proj = nn.Linear(d_model, 384)

    def forward(self, input_ids, attention_mask, labels, wsi_vec):
        bsz = input_ids.size(0)

        # ── Visual tokens ────────────────────────────────────────
        vis_tokens = self.vis_proj(wsi_vec)                     # (B, num_vis * d_model)
        vis_tokens = vis_tokens.view(bsz, self.num_vis_tokens, self.d_model)  # (B, 8, 512)

        # Get text embeddings from T5 encoder embedding layer
        text_embeds = self.t5.encoder.embed_tokens(input_ids)   # (B, L, 512)

        # Prepend visual tokens
        combined_embeds = torch.cat([vis_tokens, text_embeds], dim=1)  # (B, 8+L, 512)

        # Extend attention mask for visual tokens (all 1s)
        vis_mask = torch.ones(bsz, self.num_vis_tokens,
                              device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([vis_mask, attention_mask], dim=1)  # (B, 8+L)

        # Forward through T5 with custom encoder inputs
        encoder_outputs = self.t5.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )

        # Decoder forward (teacher forcing)
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=combined_mask,
            labels=labels,
        )

        # ── Semantic alignment head ──────────────────────────────
        # Get decoder hidden states (need to forward decoder manually for hidden)
        decoder_hidden = outputs.logits  # We need actual hidden states, not logits

        # Re-run decoder to get hidden states
        # Use the labels as decoder input (shift right is handled internally)
        decoder_out = self.t5.decoder(
            input_ids=self._shift_right(labels, self.t5.config.decoder_start_token_id),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=combined_mask,
        )
        dec_hidden = decoder_out.last_hidden_state             # (B, T, 512)

        # Mean pool over non-padding tokens
        label_mask = (labels != -100).float().unsqueeze(-1)     # (B, T, 1)
        pooled = (dec_hidden * label_mask).sum(dim=1) / (label_mask.sum(dim=1) + 1e-8)  # (B, 512)
        pred_sem = self.sem_proj(pooled)                        # (B, 384)

        return outputs.loss, pred_sem

    @staticmethod
    def _shift_right(labels, decoder_start_token_id):
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = decoder_start_token_id
        shifted[shifted == -100] = 0  # replace -100 with pad
        return shifted


# ── Semantic loss (frozen sentence-transformers) ────────────────
class SemanticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Freeze
        for p in self.st_model.parameters():
            p.requires_grad = False
        self.st_model.eval()

    @torch.no_grad()
    def encode_targets(self, texts, device):
        """Encode target texts into 384-d embeddings."""
        embs = self.st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return torch.from_numpy(embs).float().to(device)

    def forward(self, pred_emb, target_emb):
        """Cosine distance loss: 1 - cosine(pred, target)."""
        pred_emb = F.normalize(pred_emb, dim=-1)
        target_emb = F.normalize(target_emb, dim=-1)
        cosine_sim = (pred_emb * target_emb).sum(dim=-1)       # (B,)
        return (1 - cosine_sim).mean()


# ── Validation ROUGE-L ──────────────────────────────────────────
def compute_val_rouge(model, tokenizer, val_loader, device, max_len=MAX_TARGET_LEN):
    """Quick ROUGE-L on val set using greedy decoding."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    model.eval()
    all_scores = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            wsi_vec = batch["wsi_vec"].to(device)
            target_texts = batch["target_text"]
            bsz = input_ids.size(0)

            # Build encoder outputs with visual tokens
            vis_tokens = model.vis_proj(wsi_vec).view(bsz, model.num_vis_tokens, model.d_model)
            text_embeds = model.t5.encoder.embed_tokens(input_ids)
            combined_embeds = torch.cat([vis_tokens, text_embeds], dim=1)
            vis_mask = torch.ones(bsz, model.num_vis_tokens,
                                  device=device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([vis_mask, attention_mask], dim=1)

            encoder_outputs = model.t5.encoder(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
            )

            # Generate
            generated = model.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=combined_mask,
                max_length=max_len,
                num_beams=1,          # greedy for speed during training
                early_stopping=True,
            )

            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for pred, gt in zip(preds, target_texts):
                score = scorer.score(gt, pred)
                all_scores.append(score['rougeL'].fmeasure)

    model.train()
    return sum(all_scores) / len(all_scores) if all_scores else 0.0


# ── Training ────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train T5 + visual + semantic model")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (folder in runs/). Default: ours_prompt_semantic")
    parser.add_argument("--semantic_weight", type=float, default=None,
                        help="Semantic loss weight (lambda). Default: 0.2")
    args = parser.parse_args()

    # Override globals if CLI args provided
    save_dir = Path(f"runs/{args.exp_name}") if args.exp_name else SAVE_DIR
    lambda_sem = args.semantic_weight if args.semantic_weight is not None else LAMBDA_SEM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: epochs={EPOCHS}, bs={BATCH_SIZE}, lr={LR}, "
          f"lambda_sem={lambda_sem}, vis_tokens={NUM_VISUAL_TOKENS}")
    print(f"Save dir: {save_dir}")
    print("=" * 60)

    # ── Load tokenizer & model ───────────────────────────────────
    print("Loading T5-small...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = PromptSemanticModel(MODEL_NAME).to(device)
    print(f"  T5 params: {sum(p.numel() for p in model.t5.parameters()):,}")
    print(f"  Vis proj:  {sum(p.numel() for p in model.vis_proj.parameters()):,}")
    print(f"  Sem proj:  {sum(p.numel() for p in model.sem_proj.parameters()):,}")

    # ── Semantic loss ────────────────────────────────────────────
    print("Loading sentence-transformers for semantic loss...")
    sem_loss_fn = SemanticLoss()
    # Move the sentence model to device if GPU
    if device.type == "cuda":
        sem_loss_fn.st_model = sem_loss_fn.st_model.to(device)

    # ── Datasets ─────────────────────────────────────────────────
    train_ds = PromptDataset(OUTPUT_DIR / "prompt_train.jsonl", tokenizer,
                             MAX_INPUT_LEN, MAX_TARGET_LEN)
    val_ds = PromptDataset(OUTPUT_DIR / "prompt_val.jsonl", tokenizer,
                           MAX_INPUT_LEN, MAX_TARGET_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, drop_last=False)

    print(f"  Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds)} samples")

    # ── Optimizer ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )

    # ── Training loop ────────────────────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_rouge = -1
    patience_counter = 0
    train_log = []

    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_ce = 0
        epoch_sem = 0
        epoch_total = 0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            wsi_vec = batch["wsi_vec"].to(device)
            target_texts = batch["target_text"]

            # Forward
            loss_ce, pred_sem = model(input_ids, attention_mask, labels, wsi_vec)

            # Semantic loss
            target_sem = sem_loss_fn.encode_targets(target_texts, device)
            loss_sem = sem_loss_fn(pred_sem, target_sem)

            # Combined loss
            loss = loss_ce + lambda_sem * loss_sem

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_ce += loss_ce.item()
            epoch_sem += loss_sem.item()
            epoch_total += loss.item()
            n_batches += 1

        avg_ce = epoch_ce / n_batches
        avg_sem = epoch_sem / n_batches
        avg_total = epoch_total / n_batches
        elapsed = time.time() - t0

        # ── Validation ───────────────────────────────────────────
        val_rouge = compute_val_rouge(model, tokenizer, val_loader, device)

        log_entry = {
            "epoch": epoch,
            "loss_ce": round(avg_ce, 4),
            "loss_sem": round(avg_sem, 4),
            "loss_total": round(avg_total, 4),
            "val_rougeL": round(val_rouge, 4),
            "time_sec": round(elapsed, 1),
        }
        train_log.append(log_entry)

        improved = ""
        if val_rouge > best_val_rouge:
            best_val_rouge = val_rouge
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_rougeL": val_rouge,
            }, save_dir / "best.pt")
            improved = " ✅ BEST"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"CE={avg_ce:.4f} SEM={avg_sem:.4f} Total={avg_total:.4f} | "
              f"ValRL={val_rouge:.4f}{improved} | "
              f"{elapsed:.0f}s")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # ── Save training log ────────────────────────────────────────
    log_path = save_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best val ROUGE-L: {best_val_rouge:.4f}")
    print(f"  Checkpoint:       {save_dir / 'best.pt'}")
    print(f"  Log:              {log_path}")
    print(f"\nNext: python3 predict_prompt_semantic.py --split val")


if __name__ == "__main__":
    main()
