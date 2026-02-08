import pandas as pd
from .logger import setup_logger

logger = setup_logger("Evaluator")

def evaluate_results(matches, bank, reg):
    """
    Computes performance metrics based on simulated ground truth (Considering the ids corresponds to ground truth).
    """
    if matches.empty:
        print("No matches to evaluate.")
        return

    predictions = matches.copy()
    
    predictions['correct'] = predictions.apply(
        lambda x: x['bank_true_id'] == x['match_reg_true_id'] and x['bank_true_id'] != -1, 
        axis=1
    )
    
    true_positives = predictions['correct'].sum()
    false_positives = len(predictions) - true_positives
    
    valid_ids = set(bank['true_id']).intersection(set(reg['true_id']))
    if -1 in valid_ids: valid_ids.remove(-1)
    if 0 in valid_ids: valid_ids.remove(0)
    
    valid_bank_count = len(bank[bank['true_id'].isin(valid_ids)])
    false_negatives = valid_bank_count - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info("--- Performance Report ---")
    logger.info(f"Matches Proposed: {len(predictions)}")
    logger.info(f"Correct Matches:  {true_positives}")
    logger.info(f"Precision:        {precision:.2%}")
    logger.info(f"Recall:           {recall:.2%}")
    logger.info(f"F1 Score:         {f1:.2%}")
