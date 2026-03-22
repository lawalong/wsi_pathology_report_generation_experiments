import json

# Check all prediction files for TCGA-BH-A18S
files = [
    ('baseline_retrieval', 'runs/baseline_retrieval/predictions_test.jsonl'),
    ('baseline_structured', 'runs/baseline_structured/predictions_test.jsonl'),
    ('ours_prompt_semantic', 'runs/ours_prompt_semantic/predictions_test.jsonl'),
]

for label, path in files:
    print(f"\n{'='*80}")
    print(f"FILE: {label}")
    print(f"{'='*80}")
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj['case_id'] == 'TCGA-BH-A18S':
                print(f"\nGT:\n{obj['gt']}")
                print(f"\nPRED:\n{obj['pred']}")
                if 'gt_keywords' in obj:
                    print(f"\nGT Keywords: {obj['gt_keywords']}")
                if 'pred_keywords' in obj:
                    print(f"\nPred Keywords: {obj['pred_keywords']}")
                break

# Also check the raw Report.txt
print(f"\n{'='*80}")
print("RAW REPORT FILE")
print(f"{'='*80}")
try:
    with open('reference/TCGA-BRCA/TCGA-BH-A18S/Report.txt', encoding='utf-8') as f:
        print(f.read())
except:
    print("Not found or error reading")
