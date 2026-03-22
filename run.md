# In WSL terminal
cd /mnt/c/Users/lawal/Dropbox/Study/USYD/Maphil/code

# Step 1: Build training targets (DIAGNOSIS extraction)
python3 build_targets.py

# Step 2: Build structured text + keywords
python3 build_structured_text.py

# Step 3: Run baselines
python3 baseline_retrieval.py
python3 baseline_structured.py

# Step 4: Evaluate
python3 evaluate_metrics.py --exp baseline_retrieval --split test
python3 evaluate_metrics.py --exp baseline_structured --split test

# Step 5: Compare results
python3 compare_baselines.py