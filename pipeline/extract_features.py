# Step6：Feature Extraction（PyTorch + ResNet50，最稳最省事）

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms

# ============ Configuration ============
# Output folder structure (shared with other scripts)
OUTPUT_ROOT = "output"
OUTPUT_PATCHES = os.path.join(OUTPUT_ROOT, "patches")       # input: extracted patches
OUTPUT_FEATURES = os.path.join(OUTPUT_ROOT, "features")     # output: feature vectors

# Create output directories
os.makedirs(OUTPUT_FEATURES, exist_ok=True)

PATCH_DIR = OUTPUT_PATCHES
OUT_NPY = os.path.join(OUTPUT_FEATURES, "features_resnet50.npy")
OUT_PATCH_LIST = os.path.join(OUTPUT_FEATURES, "patch_list.txt")
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 1) ResNet50 去掉分类头，输出 2048-d 特征
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = nn.Identity()
resnet = resnet.to(device).eval()

# 2) 预处理：标准 ImageNet normalize（先用这个跑通）
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

paths = sorted(glob.glob(os.path.join(PATCH_DIR, "*.png")))
assert len(paths) > 0, f"No patches found in {PATCH_DIR}"
print("Num patches:", len(paths))

features = []
with torch.no_grad():
    for i in tqdm(range(0, len(paths), BATCH_SIZE)):
        batch_paths = paths[i:i+BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(tfm(img))
        x = torch.stack(imgs, dim=0).to(device)   # [B,3,224,224]
        feat = resnet(x)                          # [B,2048]
        features.append(feat.cpu().numpy())

features = np.concatenate(features, axis=0)       # [N,2048]
np.save(OUT_NPY, features)

# 同时保存 patch 路径，方便对齐和复现
with open(OUT_PATCH_LIST, "w") as f:
    for p in paths:
        f.write(p + "\n")

print("Saved:", OUT_NPY, "shape=", features.shape)
print("Saved:", OUT_PATCH_LIST)