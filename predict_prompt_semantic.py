"""
Predict with the trained Prompt-Guided Semantic model.

Loads best.pt checkpoint, generates reports for val/test,
outputs predictions compatible with evaluate_metrics.py.

Usage:
  python3 predict_prompt_semantic.py --split val
  python3 predict_prompt_semantic.py --split test
"""

import json
import re
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from transformers import T5Tokenizer

from config import OUTPUT_DIR

# Import model class from training script
from train_prompt_semantic import (
    PromptSemanticModel, PromptDataset,
    MODEL_NAME, MAX_INPUT_LEN, MAX_TARGET_LEN, NUM_VISUAL_TOKENS, SAVE_DIR
)

# ── Keyword extraction (reuse logic from build_structured_text.py) ──
MEDICAL_PATTERNS = [
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
    r'\b(\d+\.?\d*\s*[cx]m)\b',
]


def extract_keywords(text, top_k=20):
    """Extract medical keywords from generated text."""
    keywords = []
    text_lower = text.lower()
    for pattern in MEDICAL_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        keywords.extend(matches)
    counter = Counter(keywords)
    return [kw for kw, _ in counter.most_common(top_k)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--beam", type=int, default=4, help="Beam size for generation")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (folder in runs/). Default: ours_prompt_semantic")
    args = parser.parse_args()

    # Override save dir if exp_name provided
    save_dir = Path(f"runs/{args.exp_name}") if args.exp_name else SAVE_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Split:  {args.split}")
    print(f"Beam:   {args.beam}")
    print(f"Exp:    {save_dir}")
    print("=" * 60)

    # ── Load model ───────────────────────────────────────────────
    print("Loading model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = PromptSemanticModel(MODEL_NAME).to(device)

    ckpt_path = save_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val ROUGE-L = {ckpt['val_rougeL']:.4f})")

    model.eval()

    # ── Load dataset ─────────────────────────────────────────────
    prompt_file = OUTPUT_DIR / f"prompt_{args.split}.jsonl"
    if not prompt_file.exists():
        print(f"❌ Prompt file not found: {prompt_file}")
        return

    samples = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  Loaded {len(samples)} samples from {prompt_file}")

    # ── Generate ─────────────────────────────────────────────────
    out_dir = save_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{args.split}.jsonl"

    print(f"\nGenerating reports...")
    results = []

    with torch.no_grad():
        for i, s in enumerate(samples):
            # Load WSI
            wsi_path = Path(s["wsi_path"])
            if not wsi_path.is_absolute():
                wsi_path = OUTPUT_DIR.parent / wsi_path
            wsi_vec = torch.from_numpy(
                np.load(wsi_path).astype(np.float32)
            ).unsqueeze(0).to(device)

            # Tokenize prompt
            enc = tokenizer(
                s["prompt_text"],
                max_length=MAX_INPUT_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)

            # Build encoder outputs with visual tokens
            bsz = 1
            vis_tokens = model.vis_proj(wsi_vec).view(
                bsz, model.num_vis_tokens, model.d_model)
            text_embeds = model.t5.encoder.embed_tokens(input_ids)
            combined_embeds = torch.cat([vis_tokens, text_embeds], dim=1)
            vis_mask = torch.ones(bsz, model.num_vis_tokens,
                                  device=device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([vis_mask, attention_mask], dim=1)

            encoder_outputs = model.t5.encoder(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
            )

            # Generate with beam search
            generated = model.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=combined_mask,
                max_length=MAX_TARGET_LEN,
                num_beams=args.beam,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )

            pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            gt_text = s["target_text"]

            # Extract keywords
            pred_kw = extract_keywords(pred_text)
            gt_kw = extract_keywords(gt_text)

            results.append({
                "case_id": s["case_id"],
                "gt": gt_text,
                "pred": pred_text,
                "gt_keywords": gt_kw,
                "pred_keywords": pred_kw,
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                print(f"  [{i+1}/{len(samples)}] {s['case_id']}")

    # ── Save ─────────────────────────────────────────────────────
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    exp_label = args.exp_name or "ours_prompt_semantic"
    print(f"\n✅ Saved {len(results)} predictions → {pred_path}")
    print(f"\nNext: python3 evaluate_metrics.py --exp {exp_label} "
          f"--split {args.split} --bertscore")

    # ── Quick preview ────────────────────────────────────────────
    if results:
        print(f"\n── Preview: {results[0]['case_id']} ──")
        print(f"PRED: {results[0]['pred'][:300]}...")
        print(f"\nGT:   {results[0]['gt'][:300]}...")


if __name__ == "__main__":
    main()
