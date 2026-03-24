"""
Generate publication-quality heatmap from patch–concept similarity CSV.
Output: results/heatmap_TCGA-BH-A18S.png and .pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load data ────────────────────────────────────────────────────
csv_path = "results/heatmap_data_TCGA-BH-A18S.csv"
df = pd.read_csv(csv_path, index_col=0)

# Drop non-numeric helper column
if "top_concept" in df.columns:
    df = df.drop("top_concept", axis=1)

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"LN_metastasis column index: {list(df.columns).index('LN_metastasis')}")
print(f"Value range: [{df.values.min():.4f}, {df.values.max():.4f}]")

# ── Pretty column labels ─────────────────────────────────────────
pretty_labels = {
    "invasive_ductal": "Invasive\nDuctal",
    "lobular_in_situ": "Lobular\nIn Situ",
    "LN_metastasis":   "LN\nMetastasis",
    "negative_margin": "Negative\nMargin",
    "benign":          "Benign\nTissue",
    "necrosis":        "Tumor\nNecrosis",
    "normal_stroma":   "Normal\nStroma",
    "high_grade":      "High\nGrade",
}

col_labels = [pretty_labels.get(c, c) for c in df.columns]
ln_col_idx = list(df.columns).index("LN_metastasis")

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

data = df.values  # [32, 8]
n_patches, n_terms = data.shape

# Heatmap
im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest",
               vmin=0.0, vmax=0.35)

# X-axis: clinical terms
ax.set_xticks(range(n_terms))
ax.set_xticklabels(col_labels, fontsize=12, fontweight="bold", ha="center")
ax.xaxis.set_ticks_position("bottom")

# Y-axis: every 4th patch
ytick_positions = list(range(0, n_patches, 4))
ytick_labels = [f"Patch {i}" for i in ytick_positions]
ax.set_yticks(ytick_positions)
ax.set_yticklabels(ytick_labels, fontsize=11)
ax.set_ylabel("Tissue Patches", fontsize=13, fontweight="bold")

# Title
ax.set_title("Patch–Concept Semantic Similarity\n(TCGA-BH-A18S)",
             fontsize=15, fontweight="bold", pad=15)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label("Cosine Similarity", fontsize=12, fontweight="bold")
cbar.ax.tick_params(labelsize=11)

# ── Highlight LN_metastasis column with red rectangle ─────────
rect = mpatches.FancyBboxPatch(
    (ln_col_idx - 0.5, -0.5),        # bottom-left corner
    1.0,                               # width (1 column)
    n_patches,                         # height (all rows)
    linewidth=2.5,
    edgecolor="red",
    facecolor="none",
    linestyle="-",
    boxstyle="square,pad=0",
    zorder=10,
)
ax.add_patch(rect)

# Add a small red arrow annotation above the highlighted column
ax.annotate("Highest\nacross all\npatches",
            xy=(ln_col_idx, -0.5), xytext=(ln_col_idx + 2.8, -4.5),
            fontsize=9, fontweight="bold", color="red", ha="center",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

# ── Style ─────────────────────────────────────────────────────────
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Add thin grid lines between cells
for i in range(n_terms + 1):
    ax.axvline(i - 0.5, color="white", linewidth=0.5)
for i in range(n_patches + 1):
    ax.axhline(i - 0.5, color="white", linewidth=0.5)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────
png_path = "results/heatmap_TCGA-BH-A18S.png"
pdf_path = "results/heatmap_TCGA-BH-A18S.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
print(f"\n✅ Saved: {png_path}")
print(f"✅ Saved: {pdf_path}")

plt.close()
