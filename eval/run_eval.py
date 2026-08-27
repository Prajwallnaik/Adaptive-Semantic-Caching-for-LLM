import json
import os
import pandas as pd
from app.core.hybrid_search import hybrid_search
from app.cache.write_behind import write_cache_entry
from app.core.adaptive_threshold import decide, Decision
from app.core.verification import verify_match
from app.config import settings

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate_threshold(threshold, paraphrase_pairs, near_miss_pairs, band):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Evaluate paraphrases (should hit)
    for pair in paraphrase_pairs:
        # Write query1 to cache
        point_id = write_cache_entry(pair["query1"], "test_answer", user_id="eval")
        
        # Search for query2
        results = hybrid_search(pair["query2"], user_id="eval")
        if results:
            score = results[0].score
            decision = decide(score, threshold, band)
            
            if decision == Decision.HIT:
                true_positives += 1
            elif decision == Decision.VERIFY:
                # Mock verification, assume rule-based passes if they are similar enough
                # Here we just use the real verification
                if verify_match(pair["query2"], pair["query1"], "test_answer"):
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                false_negatives += 1
        else:
            false_negatives += 1

    # Evaluate near misses (should miss)
    for pair in near_miss_pairs:
        point_id = write_cache_entry(pair["query1"], "test_answer", user_id="eval")
        results = hybrid_search(pair["query2"], user_id="eval")
        if results:
            score = results[0].score
            decision = decide(score, threshold, band)
            
            if decision == Decision.HIT:
                false_positives += 1
            elif decision == Decision.VERIFY:
                if verify_match(pair["query2"], pair["query1"], "test_answer"):
                    false_positives += 1
        # If no results or miss, that's a true negative (which is correct behavior)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall

def run_eval():
    print("Running evaluation harness...")
    base_dir = os.path.dirname(__file__)
    paraphrases = load_data(os.path.join(base_dir, "datasets/paraphrase_pairs.json"))
    near_misses = load_data(os.path.join(base_dir, "datasets/near_miss_pairs.json"))

    band = settings.verification_band
    results = []

    # Requires Qdrant to be running (either via docker-compose or testcontainers)
    # Ensure qdrant is up for this eval to work
    
    thresholds = [x / 100.0 for x in range(70, 96)]
    for t in thresholds:
        print(f"Evaluating threshold {t:.2f}...")
        p, r = evaluate_threshold(t, paraphrases, near_misses, band)
        results.append({"threshold": t, "precision": p, "recall": r})

    df = pd.DataFrame(results)
    
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    df.to_csv(os.path.join(base_dir, "results/eval_results.csv"), index=False)
    
    print("\nEvaluation Results:")
    print(df.to_string())
    
    # Recommend optimal threshold (e.g. highest F1 score)
    df['f1'] = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
    df['f1'] = df['f1'].fillna(0)
    best_row = df.loc[df['f1'].idxmax()]
    print(f"\nRecommended Threshold based on F1 Score: {best_row['threshold']:.2f}")

if __name__ == "__main__":
    run_eval()
