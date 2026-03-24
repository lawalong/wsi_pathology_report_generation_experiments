"""
Generate Semantic Alignment Heatmap Data

Computes a (patches × clinical_terms) similarity matrix for PPT visualization.

Approach:
  - Patch features: ResNet50 [N, 2048] — already extracted
  - Text features: Use the SAME all-MiniLM-L6-v2 sentence encoder from our
    semantic loss, but project both into a shared space via our trained
    semantic projection head (sem_proj: 512 → 384).
  - To bridge ResNet→MiniLM, we use our trained visual projection
    (vis_proj: 2048 → 8×512) then mean-pool the 8 tokens → [512],
    then sem_proj → [384].
  - Clinical terms are encoded directly by MiniLM → [384].
  - Cosine similarity between each patch embedding and each term.

This is faithful to our actual model — the projection heads are loaded
from the trained checkpoint.

Output:
  results/heatmap_data_{case_id}.json
  results/heatmap_data_{case_id}.csv
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Import our model class to load checkpoint
from train_prompt_semantic import PromptSemanticModel, MODEL_NAME, SAVE_DIR

# ── Config ───────────────────────────────────────────────────────
CASE_ID = "TCGA-BH-A18S"
FEATURES_DIR = Path("data/features")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Clinical terms — chosen to span the concept space in our evaluation
CLINICAL_TERMS = [
    "invasive ductal carcinoma",
    "lobular carcinoma in situ",
    "lymph node metastasis",
    "negative surgical margins",
    "benign breast tissue",
    "tumor necrosis",
    "normal stroma",
    "high-grade nuclear atypia",
]

SHORT_LABELS = [
    "invasive_ductal",
    "lobular_in_situ",
    "LN_metastasis",
    "negative_margin",
    "benign",
    "necrosis",
    "normal_stroma",
    "high_grade",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Case:   {CASE_ID}")
    print(f"Terms:  {len(CLINICAL_TERMS)}")

    # ── Load patch features ──────────────────────────────────────
    feat_path = FEATURES_DIR / f"{CASE_ID}.npy"
    patch_features = np.load(feat_path).astype(np.float32)  # [N, 2048]
    n_patches = patch_features.shape[0]
    print(f"Patches: {n_patches} × {patch_features.shape[1]}")

    # ── Load trained model (for projection heads) ────────────────
    print("Loading trained model...")
    model = PromptSemanticModel(MODEL_NAME).to(device)
    ckpt = torch.load(SAVE_DIR / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}")

    # ── Load sentence encoder (same as semantic loss) ────────────
    print("Loading sentence encoder...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    st_model.eval()

    # ── Encode clinical terms → [K, 384] ────────────────────────
    term_embs = st_model.encode(CLINICAL_TERMS, convert_to_numpy=True,
                                show_progress_bar=False)
    term_embs = torch.from_numpy(term_embs).float().to(device)  # [K, 384]
    term_embs = F.normalize(term_embs, dim=-1)
    print(f"Term embeddings: {term_embs.shape}")

    # ── Project patch features through visual + semantic heads ───
    # vis_proj: 2048 → (8 × 512)
    # We mean-pool the 8 visual tokens → [512]
    # sem_proj: 512 → 384
    print("Projecting patch features...")
    with torch.no_grad():
        patch_tensor = torch.from_numpy(patch_features).to(device)  # [N, 2048]

        # Visual projection → [N, 8, 512]
        vis_out = model.vis_proj(patch_tensor)  # [N, 8*512]
        vis_out = vis_out.view(n_patches, model.num_vis_tokens, model.d_model)  # [N, 8, 512]

        # Mean pool over 8 tokens → [N, 512]
        vis_pooled = vis_out.mean(dim=1)  # [N, 512]

        # Semantic projection → [N, 384]
        patch_sem = model.sem_proj(vis_pooled)  # [N, 384]
        patch_sem = F.normalize(patch_sem, dim=-1)

    print(f"Patch semantic embeddings: {patch_sem.shape}")

    # ── Compute cosine similarity matrix ─────────────────────────
    # [N, 384] × [384, K] → [N, K]
    sim_matrix = (patch_sem @ term_embs.T).cpu().numpy()  # [N, K]
    print(f"Similarity matrix: {sim_matrix.shape}")
    print(f"  Range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")
    print(f"  Mean:  {sim_matrix.mean():.4f}")

    # ── Save as JSON ─────────────────────────────────────────────
    output = {
        "case_id": CASE_ID,
        "n_patches": n_patches,
        "clinical_terms": CLINICAL_TERMS,
        "short_labels": SHORT_LABELS,
        "explanation": (
            "Each row is a tissue patch from the WSI. "
            "Each column is a clinical concept. "
            "Values are cosine similarities in the learned semantic space "
            "(projected through our trained vis_proj + sem_proj heads). "
            "Higher values indicate the patch embedding is closer to that "
            "clinical concept in the shared semantic space. "
            "This reflects what the model 'sees' in each patch."
        ),
        "similarity_matrix": sim_matrix.round(4).tolist(),
        "per_patch_summary": [],
    }

    # Add per-patch top concept
    for i in range(n_patches):
        row = sim_matrix[i]
        top_idx = int(row.argmax())
        output["per_patch_summary"].append({
            "patch_idx": i,
            "top_concept": SHORT_LABELS[top_idx],
            "top_score": round(float(row[top_idx]), 4),
            "scores": {lbl: round(float(row[j]), 4) for j, lbl in enumerate(SHORT_LABELS)},
        })

    json_path = RESULTS_DIR / f"heatmap_data_{CASE_ID}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved: {json_path}")

    # ── Save as CSV (for easy Excel/plotting) ────────────────────
    csv_path = RESULTS_DIR / f"heatmap_data_{CASE_ID}.csv"
    with open(csv_path, "w") as f:
        f.write("patch_idx," + ",".join(SHORT_LABELS) + ",top_concept\n")
        for i in range(n_patches):
            row = sim_matrix[i]
            top_idx = int(row.argmax())
            vals = ",".join(f"{v:.4f}" for v in row)
            f.write(f"{i},{vals},{SHORT_LABELS[top_idx]}\n")
    print(f"✅ Saved: {csv_path}")

    # ── Print preview ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  HEATMAP PREVIEW — {CASE_ID} ({n_patches} patches × {len(CLINICAL_TERMS)} terms)")
    print(f"{'='*70}")
    header = f"{'Patch':>6} " + " ".join(f"{lbl:>14}" for lbl in SHORT_LABELS) + "  TOP"
    print(header)
    print("-" * len(header))

    for i in range(min(n_patches, 15)):  # show first 15
        row = sim_matrix[i]
        top_idx = int(row.argmax())
        vals = " ".join(f"{v:>14.4f}" for v in row)
        print(f"{i:>6} {vals}  {SHORT_LABELS[top_idx]}")

    if n_patches > 15:
        print(f"  ... ({n_patches - 15} more patches)")

    # ── Aggregate statistics ─────────────────────────────────────
    print(f"\nMean similarity per concept:")
    for j, lbl in enumerate(SHORT_LABELS):
        col = sim_matrix[:, j]
        print(f"  {lbl:<16} mean={col.mean():.4f}  max={col.max():.4f}  "
              f"min={col.min():.4f}")


if __name__ == "__main__":
    main()
