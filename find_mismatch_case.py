"""Find test cases where retrieval has semantic mismatch with GT — for PPT slide."""
import json, re
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def get_key_concepts(text):
    t = text.lower()
    c = []
    if 'ductal' in t: c.append('ductal')
    if 'lobular' in t: c.append('lobular')
    if 'invasive' in t or 'infiltrating' in t: c.append('invasive')
    if 'malignant' in t or 'malignancy' in t: c.append('malignant')
    if 'benign' in t: c.append('benign')
    if 'carcinoma' in t: c.append('carcinoma')
    if 'metasta' in t: c.append('metastatic')
    if 'negative' in t: c.append('negative')
    if 'positive' in t: c.append('positive')
    if re.search(r'grade\s*[123iIvV]', t): c.append('grade')
    if 'inflammation' in t or 'inflammatory' in t: c.append('inflammation')
    if 'in situ' in t or 'dcis' in t: c.append('in_situ')
    if 'lymph node' in t: c.append('lymph_node')
    if 'margin' in t: c.append('margin')
    return set(c)

# Load predictions
with open('runs/baseline_structured/predictions_test.jsonl') as f:
    struct_preds = [json.loads(l) for l in f]
with open('runs/ours_prompt_semantic/predictions_test.jsonl') as f:
    ours_preds = [json.loads(l) for l in f]
ours_map = {p['case_id']: p for p in ours_preds}

# Also load topk_fusion for retrieved report
with open('runs/topk_fusion_k3/predictions_test.jsonl') as f:
    fusion_preds = [json.loads(l) for l in f]
fusion_map = {p['case_id']: p for p in fusion_preds}

interesting = []
for sp in struct_preds:
    cid = sp['case_id']
    gt = sp['gt']
    pred_struct = sp['pred']
    op = ours_map.get(cid)
    if not op: continue
    pred_ours = op['pred']

    rl_struct = scorer.score(gt, pred_struct)['rougeL'].fmeasure
    rl_ours = scorer.score(gt, pred_ours)['rougeL'].fmeasure

    gt_c = get_key_concepts(gt)
    struct_c = get_key_concepts(pred_struct)
    ours_c = get_key_concepts(pred_ours)

    missed_by_struct = gt_c - struct_c
    wrong_in_struct = struct_c - gt_c
    missed_by_ours = gt_c - ours_c

    if missed_by_struct or wrong_in_struct:
        interesting.append({
            'cid': cid, 'rl_s': rl_struct, 'rl_o': rl_ours,
            'gt_c': gt_c, 'struct_c': struct_c, 'ours_c': ours_c,
            'missed': missed_by_struct, 'wrong': wrong_in_struct,
            'missed_o': missed_by_ours,
            'gt': gt, 'pred_struct': pred_struct, 'pred_ours': pred_ours,
            'score': len(missed_by_struct) + len(wrong_in_struct),
        })

interesting.sort(key=lambda x: x['score'], reverse=True)

print("=" * 100)
print("TOP 10 MISMATCH CASES (retrieval misses or adds wrong concepts vs GT)")
print("=" * 100)
for item in interesting[:10]:
    print(f"\n{'='*80}")
    print(f"CASE: {item['cid']}  |  RL_struct={item['rl_s']:.4f}  RL_ours={item['rl_o']:.4f}")
    print(f"  GT concepts:     {item['gt_c']}")
    print(f"  Struct concepts: {item['struct_c']}")
    print(f"    MISSED by retrieval: {item['missed']}")
    print(f"    WRONG in retrieval:  {item['wrong']}")
    print(f"  Ours concepts:   {item['ours_c']}")
    print(f"    MISSED by ours:     {item['missed_o']}")
    print(f"\n  --- GT (first 300 chars) ---")
    print(f"  {item['gt'][:300]}")
    print(f"\n  --- Structured retrieval (first 300 chars) ---")
    print(f"  {item['pred_struct'][:300]}")
    print(f"\n  --- Ours model output (first 300 chars) ---")
    print(f"  {item['pred_ours'][:300]}")
