import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""Quick fix: Deduplicate dataset.jsonl and rebuild splits. NO LEAKAGE."""
import json, random
from config import DATASET_FILE, SPLITS_FILE, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED

# Read existing dataset
items = [json.loads(l) for l in open(DATASET_FILE)]
print(f"Before dedup: {len(items)} entries")

# Dedup: keep first entry per case_id
seen = {}
for it in items:
    cid = it["case_id"]
    if cid not in seen:
        seen[cid] = it
    else:
        print(f"  ⚠️  Dropping duplicate: {cid}")

dataset = list(seen.values())
print(f"After dedup:  {len(dataset)} unique cases")

# Write deduped dataset
with open(DATASET_FILE, "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"✅ Saved: {DATASET_FILE}")

# Rebuild splits on unique case_ids
random.seed(RANDOM_SEED)
case_ids = [e["case_id"] for e in dataset]
random.shuffle(case_ids)

n = len(case_ids)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

splits = {
    "train": case_ids[:n_train],
    "val": case_ids[n_train:n_train + n_val],
    "test": case_ids[n_train + n_val:]
}

# Verify ZERO overlap
train_s = set(splits["train"])
val_s = set(splits["val"])
test_s = set(splits["test"])

tv = train_s & val_s
tt = train_s & test_s
vt = val_s & test_s

assert len(tv) == 0, f"LEAK train-val: {tv}"
assert len(tt) == 0, f"LEAK train-test: {tt}"
assert len(vt) == 0, f"LEAK val-test: {vt}"

with open(SPLITS_FILE, "w", encoding="utf-8") as f:
    json.dump(splits, f, indent=2)

print(f"✅ Saved: {SPLITS_FILE}")
print(f"\nTrain: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
print(f"Overlap: train-val={len(tv)}, train-test={len(tt)}, val-test={len(vt)}")
print("✅ DONE - NO LEAKAGE")
