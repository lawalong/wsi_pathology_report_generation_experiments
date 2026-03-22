import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Batch pipeline: Extract patches, features, and WSI-level features for all cases.
Reads from: data/dataset.jsonl
Outputs to:
  - data/features/{case_id}.npy  [N, 2048]
  - data/wsi/{case_id}.npy       [2048]
  - data/masks/{case_id}.png     (tissue mask)

Supports incremental processing - tracks completed cases in progress.jsonl
Run again to process only new/pending cases.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms

try:
    import openslide
except ImportError:
    print("Please install openslide-python: pip install openslide-python")
    exit(1)

# Import shared configuration
from config import (
    DATASET_FILE, FEATURES_DIR, WSI_DIR, MASKS_DIR,
    BATCH_SIZE, ProgressTracker, print_config
)

# Alias for backward compatibility
OUTPUT_FEATURES = FEATURES_DIR
OUTPUT_WSI = WSI_DIR
OUTPUT_MASKS = MASKS_DIR

# Feature extraction settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============ Initialize ResNet50 ============
print(f"Device: {DEVICE}")
print("Loading ResNet50...")
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE).eval()

# ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_patches_from_svs(svs_path: str, num_patches: int, patch_level: int, 
                              patch_size: int, min_tissue_ratio: float = 0.6):
    """
    Extract tissue patches from SVS file.
    Returns: list of PIL Images
    """
    slide = openslide.OpenSlide(svs_path)
    
    # Use level2 for tissue mask (or highest available level)
    thumb_level = min(2, slide.level_count - 1)
    
    # Get thumbnail for tissue detection
    w2, h2 = slide.level_dimensions[thumb_level]
    thumb = slide.read_region((0, 0), thumb_level, (w2, h2)).convert("RGB")
    thumb_np = np.array(thumb)
    
    # Simple white-background filtering
    bg = (thumb_np[..., 0] > 220) & (thumb_np[..., 1] > 220) & (thumb_np[..., 2] > 220)
    mask = ~bg  # True = tissue
    
    # Get tissue coordinates
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [], mask
    
    coords = list(zip(xs, ys))
    random.shuffle(coords)
    
    # Calculate scale factors
    ds_patch = slide.level_downsamples[patch_level]
    ds_thumb = slide.level_downsamples[thumb_level]
    scale = ds_thumb / ds_patch
    
    w1, h1 = slide.level_dimensions[patch_level]
    
    patches = []
    tries = 0
    max_tries = num_patches * 50
    
    while len(patches) < num_patches and tries < max_tries:
        tries += 1
        x2, y2 = random.choice(coords)
        
        # Convert to patch level coordinates
        x1 = int(x2 * scale)
        y1 = int(y2 * scale)
        
        # Get top-left corner
        x1_tl = x1 - patch_size // 2
        y1_tl = y1 - patch_size // 2
        
        # Boundary check
        if x1_tl < 0 or y1_tl < 0 or x1_tl + patch_size >= w1 or y1_tl + patch_size >= h1:
            continue
        
        # Convert to level0 coordinates
        x0 = int(x1_tl * ds_patch)
        y0 = int(y1_tl * ds_patch)
        
        patch = slide.read_region((x0, y0), patch_level, (patch_size, patch_size)).convert("RGB")
        patch_np = np.array(patch)
        
        # Check tissue ratio
        patch_bg = (patch_np[..., 0] > 220) & (patch_np[..., 1] > 220) & (patch_np[..., 2] > 220)
        tissue_ratio = 1.0 - patch_bg.mean()
        
        if tissue_ratio < min_tissue_ratio:
            continue
        
        patches.append(patch)
    
    slide.close()
    return patches, mask


def extract_features_from_patches(patches: list) -> np.ndarray:
    """
    Extract ResNet50 features from patches.
    Returns: np.ndarray of shape [N, 2048]
    """
    if len(patches) == 0:
        return np.zeros((0, 2048), dtype=np.float32)
    
    features = []
    
    with torch.no_grad():
        for i in range(0, len(patches), BATCH_SIZE):
            batch = patches[i:i + BATCH_SIZE]
            imgs = torch.stack([transform(img) for img in batch]).to(DEVICE)
            feat = resnet(imgs)
            features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def process_case(case: dict, tracker: ProgressTracker = None) -> bool:
    """
    Process a single case: extract patches → features → WSI feature.
    Returns: True if successful, False otherwise.
    """
    case_id = case["case_id"]
    svs_path = case["svs_path"]
    num_patches = case.get("num_patches", 50)
    patch_level = case.get("patch_level", 1)
    patch_size = case.get("patch_size", 256)
    
    # Output paths
    feature_path = OUTPUT_FEATURES / f"{case_id}.npy"
    wsi_path = OUTPUT_WSI / f"{case_id}.npy"
    mask_path = OUTPUT_MASKS / f"{case_id}.png"
    
    # Skip if already completed (tracked in progress.jsonl)
    if tracker and tracker.is_completed(case_id):
        return True
    
    # Also skip if output files exist (backward compatibility)
    if feature_path.exists() and wsi_path.exists():
        if tracker:
            tracker.mark_completed(case_id, num_patches=num_patches)
        return True
    
    # Check SVS file exists
    if not os.path.exists(svs_path):
        error_msg = f"SVS file not found: {svs_path}"
        print(f"  ❌ {error_msg}")
        if tracker:
            tracker.mark_failed(case_id, error_msg)
        return False
    
    try:
        # Step 1: Extract patches
        patches, mask = extract_patches_from_svs(
            svs_path, num_patches, patch_level, patch_size
        )
        
        if len(patches) == 0:
            error_msg = "No valid patches extracted"
            print(f"  ❌ {error_msg}")
            if tracker:
                tracker.mark_failed(case_id, error_msg)
            return False
        
        # Save tissue mask
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_img.save(mask_path)
        
        # Step 2: Extract features
        features = extract_features_from_patches(patches)
        np.save(feature_path, features)
        
        # Step 3: Aggregate to WSI-level (mean pooling)
        wsi_feature = features.mean(axis=0)
        np.save(wsi_path, wsi_feature)
        
        # Mark as completed
        if tracker:
            tracker.mark_completed(case_id, 
                                   num_patches=len(patches),
                                   feature_dim=features.shape[1])
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ Error: {error_msg}")
        if tracker:
            tracker.mark_failed(case_id, error_msg)
        return False


def main():
    print_config()
    
    # Create output directories
    OUTPUT_FEATURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_WSI.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)
    
    # Initialize progress tracker
    tracker = ProgressTracker()
    progress_summary = tracker.summary()
    print(f"Progress tracker: {progress_summary['completed']} completed, {progress_summary['failed']} failed")
    
    # Load dataset
    if not DATASET_FILE.exists():
        print(f"❌ Dataset file not found: {DATASET_FILE}")
        print("   Run build_index.py first to generate dataset.jsonl")
        return
    
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
    
    # Filter to pending cases only
    all_case_ids = [case["case_id"] for case in dataset]
    pending_ids = set(tracker.get_pending(all_case_ids))
    pending_dataset = [case for case in dataset if case["case_id"] in pending_ids]
    
    print(f"\nLoaded {len(dataset)} total cases from {DATASET_FILE}")
    print(f"Already completed: {len(dataset) - len(pending_dataset)}")
    print(f"Pending to process: {len(pending_dataset)}")
    
    if len(pending_dataset) == 0:
        print("\n✅ All cases already processed! Nothing to do.")
        return
    
    print(f"\nOutput: {OUTPUT_FEATURES}, {OUTPUT_WSI}, {OUTPUT_MASKS}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Process only pending cases
    success_count = 0
    fail_count = 0
    
    for case in tqdm(pending_dataset, desc="Processing cases"):
        case_id = case["case_id"]
        tqdm.write(f"\n📁 {case_id}")
        
        if process_case(case, tracker=tracker):
            tqdm.write(f"  ✅ Done: {case_id}")
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Processed this run:  {success_count + fail_count}")
    print(f"  Successful:        {success_count}")
    print(f"  Failed:            {fail_count}")
    
    # Overall progress
    overall = tracker.summary()
    print(f"\nOVERALL PROGRESS:")
    print(f"  Total completed:   {overall['completed']}")
    print(f"  Total failed:      {overall['failed']}")
    print(f"  Remaining:         {len(dataset) - overall['completed'] - overall['failed']}")
    
    print(f"\nOutputs:")
    print(f"  Features: {OUTPUT_FEATURES}/<case_id>.npy  [N, 2048]")
    print(f"  WSI:      {OUTPUT_WSI}/<case_id>.npy       [2048]")
    print(f"  Masks:    {OUTPUT_MASKS}/<case_id>.png")
    print(f"  Progress: {tracker.progress_file}")


if __name__ == "__main__":
    main()
