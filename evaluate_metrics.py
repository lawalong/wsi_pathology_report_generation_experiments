"""
Evaluate predictions using multiple metrics.
Reads: runs/<exp_name>/predictions_<split>.jsonl
Outputs: runs/<exp_name>/metrics_<split>.json

Metrics:
- ROUGE-1, ROUGE-2, ROUGE-L (F1 scores)
- BERTScore (if installed)
- Keyword Coverage (semantic alignment metric)
"""

import json
import argparse
from pathlib import Path

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Please install rouge-score: pip install rouge-score")
    exit(1)

# Optional: BERTScore
BERT_SCORE_AVAILABLE = False
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    pass


def compute_keyword_coverage(predictions: list) -> dict:
    """
    Compute keyword coverage: % of GT keywords found in prediction.
    Measures semantic alignment between predicted and ground truth.
    """
    coverages = []
    
    for item in predictions:
        gt_keywords = item.get('gt_keywords', [])
        pred_text = item.get('pred', '').lower()
        
        if not gt_keywords:
            continue
        
        # Count how many GT keywords appear in prediction
        covered = 0
        for kw in gt_keywords:
            if kw.lower() in pred_text:
                covered += 1
        
        coverage = covered / len(gt_keywords)
        coverages.append(coverage)
    
    if coverages:
        return {
            'keyword_coverage': round(sum(coverages) / len(coverages), 4),
            'keyword_coverage_min': round(min(coverages), 4),
            'keyword_coverage_max': round(max(coverages), 4)
        }
    return {'keyword_coverage': 0.0}


def compute_bert_score(predictions: list) -> dict:
    """Compute BERTScore F1 if available."""
    if not BERT_SCORE_AVAILABLE:
        return {'bertscore_available': False}
    
    preds = [item.get('pred', '') for item in predictions]
    gts = [item.get('gt', '') for item in predictions]
    
    # Filter out empty pairs
    valid = [(p, g) for p, g in zip(preds, gts) if p and g]
    if not valid:
        return {'bertscore': 0.0}
    
    preds, gts = zip(*valid)
    
    try:
        P, R, F1 = bert_score(list(preds), list(gts), lang='en', verbose=False)
        return {
            'bertscore_precision': round(P.mean().item(), 4),
            'bertscore_recall': round(R.mean().item(), 4),
            'bertscore_f1': round(F1.mean().item(), 4)
        }
    except Exception as e:
        return {'bertscore_error': str(e)}


def compute_rouge(predictions: list) -> dict:
    """
    Compute ROUGE scores for a list of predictions.
    Each prediction should have 'pred' and 'gt' fields.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for item in predictions:
        pred = item.get('pred', '')
        gt = item.get('gt', '')
        
        if not pred or not gt:
            continue
        
        result = scorer.score(gt, pred)
        
        scores['rouge1'].append(result['rouge1'].fmeasure)
        scores['rouge2'].append(result['rouge2'].fmeasure)
        scores['rougeL'].append(result['rougeL'].fmeasure)
    
    # Compute averages
    metrics = {}
    for key, values in scores.items():
        if values:
            metrics[key] = round(sum(values) / len(values), 4)
        else:
            metrics[key] = 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions using ROUGE')
    parser.add_argument('--exp', type=str, default='baseline_retrieval',
                        help='Experiment name (folder in runs/)')
    parser.add_argument('--split', type=str, default='test',
                        help='Split to evaluate (test or val)')
    parser.add_argument('--bertscore', action='store_true',
                        help='Compute BERTScore (slow, requires bert-score)')
    args = parser.parse_args()
    
    # Paths
    exp_dir = Path(f"runs/{args.exp}")
    pred_file = exp_dir / f"predictions_{args.split}.jsonl"
    metrics_file = exp_dir / f"metrics_{args.split}.json"
    
    # Check files exist
    if not pred_file.exists():
        print(f"❌ Predictions file not found: {pred_file}")
        return
    
    # Load predictions
    predictions = []
    with pred_file.open("r", encoding="utf-8") as f:
        for line in f:
            predictions.append(json.loads(line))
    
    print(f"Loaded {len(predictions)} predictions from {pred_file}")
    print("=" * 60)
    
    # Compute ROUGE
    print("Computing ROUGE scores...")
    metrics = compute_rouge(predictions)
    
    # Compute keyword coverage
    print("Computing keyword coverage...")
    kw_metrics = compute_keyword_coverage(predictions)
    metrics.update(kw_metrics)
    
    # Compute BERTScore if requested
    if args.bertscore:
        print("Computing BERTScore (this may take a while)...")
        bert_metrics = compute_bert_score(predictions)
        metrics.update(bert_metrics)
    
    # Display results
    print("\n" + "=" * 60)
    print(f"RESULTS ({args.split} set)")
    print("=" * 60)
    print(f"  ROUGE-1:          {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2:          {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L:          {metrics['rougeL']:.4f}")
    if 'keyword_coverage' in metrics:
        print(f"  Keyword Coverage: {metrics['keyword_coverage']:.4f}")
    if 'bertscore_f1' in metrics:
        print(f"  BERTScore F1:     {metrics['bertscore_f1']:.4f}")
    
    # Save metrics
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Saved metrics: {metrics_file}")
    
    # Also show per-sample results
    print("\n" + "-" * 60)
    print("Per-sample scores:")
    print("-" * 60)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for item in predictions:
        case_id = item.get('case_id', 'unknown')
        pred = item.get('pred', '')
        gt = item.get('gt', '')
        nn_case = item.get('nn_case', '')
        sim = item.get('similarity', 0)
        gt_keywords = item.get('gt_keywords', [])
        
        result = scorer.score(gt, pred)
        rouge_l = result['rougeL'].fmeasure
        
        # Per-sample keyword coverage
        if gt_keywords:
            covered = sum(1 for kw in gt_keywords if kw.lower() in pred.lower())
            kw_cov = covered / len(gt_keywords)
        else:
            kw_cov = 0.0
        
        print(f"  {case_id} → {nn_case} (sim={sim:.4f}) | ROUGE-L={rouge_l:.4f} | KW={kw_cov:.2f}")


if __name__ == "__main__":
    main()
