import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import os
import random
import numpy as np
from PIL import Image
import openslide

# Import shared configuration
from config import SVS_DIR, print_config


# ============ Configuration ============
# Find first SVS file in dataset for testing
def find_first_svs(root):
    """Find first SVS file in the dataset folder."""
    if not os.path.exists(root):
        return None
    for case_folder in os.listdir(root):
        case_path = os.path.join(root, case_folder)
        if os.path.isdir(case_path):
            for f in os.listdir(case_path):
                if f.endswith('.svs'):
                    return os.path.join(case_path, f)
    return None

# Try to find SVS file automatically
svs_path = find_first_svs(SVS_DIR)
if svs_path is None:
    print(f"❌ No SVS files found in {SVS_DIR}")
    exit(1)

print_config()
print(f"\nTest SVS: {svs_path}")
print(f"File exists: {os.path.exists(svs_path)}")

# Output folder structure
OUTPUT_ROOT = "output"
OUTPUT_DEBUG = os.path.join(OUTPUT_ROOT, "debug")           # debug images (patches, etc.)
OUTPUT_MASKS = os.path.join(OUTPUT_ROOT, "masks")           # tissue masks
OUTPUT_PATCHES = os.path.join(OUTPUT_ROOT, "patches")       # extracted patches

# Create output directories
for folder in [OUTPUT_ROOT, OUTPUT_DEBUG, OUTPUT_MASKS, OUTPUT_PATCHES]:
    os.makedirs(folder, exist_ok=True)

slide = openslide.OpenSlide(svs_path)

print("Level count:", slide.level_count)
print("Level dimensions:", slide.level_dimensions)    # 每层宽高
print("Level downsamples:", slide.level_downsamples)  # 每层相对 level0 的缩小倍数

# 1 一些常见信息
print("Objective Power:", slide.properties.get("openslide.objective-power"))
print("MPP X:", slide.properties.get("openslide.mpp-x"))
print("MPP Y:", slide.properties.get("openslide.mpp-y"))


# 2. Step 3 - 你这张的最佳选择：downsample=4 的那层（通常是 level 1）
TARGET_DOWNSAMPLE = 4

ds = list(slide.level_downsamples)
best_level = min(range(len(ds)), key=lambda i: abs(ds[i] - TARGET_DOWNSAMPLE))

print("Picked level:", best_level)
print("Picked downsample:", slide.level_downsamples[best_level])
print("Picked dims:", slide.level_dimensions[best_level])

# 3. Step4 - 读一个 patch 并保存（验证读对了）
patch_size = 256

level = 1  # 你这张固定用 1 就行（ds=4）
ds = slide.level_downsamples[level]
w_lv, h_lv = slide.level_dimensions[level]

# 取 level1 的中心点附近（大概率还是背景，没关系，先验证流程）
x_lv = w_lv // 2
y_lv = h_lv // 2

# 转到 level0 坐标
x0 = int(x_lv * ds)
y0 = int(y_lv * ds)

patch = slide.read_region((x0, y0), level, (patch_size, patch_size)).convert("RGB")
patch.save(os.path.join(OUTPUT_DEBUG, "debug_patch_level1.png"))

print("Saved debug_patch_level1.png at level", level, "ds", ds, "x0,y0=", x0, y0)


# 5. 生成 tissue mask + 从组织区域采样 patch

# 你这张：level1 用来切 patch；level2 用来做快速组织mask
patch_level = 1
thumb_level = 2

patch_size = 256
num_patches = 50          # 先少量试跑，之后再加
min_tissue_ratio = 0.6    # patch 内至少 60% 是组织（避免边缘/白底）

# ---- 5.1) 取缩略图（level2）并转 numpy ----
w2, h2 = slide.level_dimensions[thumb_level]
thumb = slide.read_region((0, 0), thumb_level, (w2, h2)).convert("RGB")
thumb_np = np.array(thumb)

# ---- 5.2) 简单白底过滤：亮度高且接近白色 → 背景 ----
# 这里用"RGB都很高"近似判白底
bg = (thumb_np[..., 0] > 220) & (thumb_np[..., 1] > 220) & (thumb_np[..., 2] > 220)
mask = ~bg  # True=组织

# 可选：存一张mask可视化检查
mask_img = (mask.astype(np.uint8) * 255)
Image.fromarray(mask_img).save(os.path.join(OUTPUT_MASKS, "tissue_mask_level2.png"))
print("Saved tissue_mask_level2.png")

# ---- 5.3) 从mask里采样点，然后回到 level1 切 patch ----
ds_patch = slide.level_downsamples[patch_level]
ds_thumb = slide.level_downsamples[thumb_level]
scale = ds_thumb / ds_patch  # 从 level2 坐标 → level1 坐标 的比例（这里=16/4=4）

ys, xs = np.where(mask)
coords = list(zip(xs, ys))
random.shuffle(coords)

saved = 0
tries = 0
max_tries = num_patches * 50  # 防止死循环

while saved < num_patches and tries < max_tries:
    tries += 1
    x2, y2 = random.choice(coords)  # level2 坐标

    # 转到 level1 坐标
    x1 = int(x2 * scale)
    y1 = int(y2 * scale)

    # 让 patch 居中：换成左上角坐标（level1）
    x1_tl = x1 - patch_size // 2
    y1_tl = y1 - patch_size // 2

    # 边界检查（level1）
    w1, h1 = slide.level_dimensions[patch_level]
    if x1_tl < 0 or y1_tl < 0 or x1_tl + patch_size >= w1 or y1_tl + patch_size >= h1:
        continue

    # level1 → level0 坐标（read_region 用 level0 坐标）
    x0 = int(x1_tl * ds_patch)
    y0 = int(y1_tl * ds_patch)

    patch = slide.read_region((x0, y0), patch_level, (patch_size, patch_size)).convert("RGB")
    patch_np = np.array(patch)

    # 计算 patch 内组织比例（同样用白底阈值）
    patch_bg = (patch_np[..., 0] > 220) & (patch_np[..., 1] > 220) & (patch_np[..., 2] > 220)
    tissue_ratio = 1.0 - patch_bg.mean()

    if tissue_ratio < min_tissue_ratio:
        continue

    patch.save(os.path.join(OUTPUT_PATCHES, f"patch_{saved:04d}.png"))
    saved += 1

print(f"Done. Saved {saved}/{num_patches} patches into {OUTPUT_PATCHES}")