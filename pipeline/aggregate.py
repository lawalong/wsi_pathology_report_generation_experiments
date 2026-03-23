# Step7：把 patch-level feature 聚合成 WSI-level（先用 mean pooling 跑通）

import os
import numpy as np

# ============ Configuration ============
# Output folder structure (shared with other scripts)
OUTPUT_ROOT = "output"
OUTPUT_FEATURES = os.path.join(OUTPUT_ROOT, "features")     # input: feature vectors
OUTPUT_WSI = os.path.join(OUTPUT_ROOT, "wsi")               # output: WSI-level features

# Create output directories
os.makedirs(OUTPUT_WSI, exist_ok=True)

INPUT_FEATURES = os.path.join(OUTPUT_FEATURES, "features_resnet50.npy")
OUT_WSI_FEATURE = os.path.join(OUTPUT_WSI, "wsi_feature_mean.npy")

feat = np.load(INPUT_FEATURES)           # [N,2048]
wsi_feat = feat.mean(axis=0)             # [2048]
np.save(OUT_WSI_FEATURE, wsi_feat)
print("Saved", OUT_WSI_FEATURE, "shape=", wsi_feat.shape)