import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    MIN_CONFIDENCE, MAX_AMOUNT_DIFF_PERCENT,
    DATE_LOOKBACK_DAYS, DATE_LOOKAHEAD_DAYS, ANN_CANDIDATES
)

class MLMatchingEngine:
    def __init__(self, bank_df, reg_df, bank_vectors, reg_vectors):
        self.bank = bank_df
        self.reg = reg_df
        self.bank_vectors = bank_vectors
        self.reg_vectors = reg_vectors
        
        # Pre-sort Register amounts
        self.reg_sorted_indices = np.argsort(self.reg['amount'].values)
        self.reg_amounts_sorted = self.reg['amount'].values[self.reg_sorted_indices]

    def _get_amount_candidates(self, target_amount):
        if target_amount <= 0: return []
        min_amt = target_amount * (1 - MAX_AMOUNT_DIFF_PERCENT)
        max_amt = target_amount * (1 + MAX_AMOUNT_DIFF_PERCENT)
        start = np.searchsorted(self.reg_amounts_sorted, min_amt, side='left')
        end = np.searchsorted(self.reg_amounts_sorted, max_amt, side='right')
        return self.reg_sorted_indices[start:end]

    def _normalize_type(self, t):
        t = str(t).upper()
        if t in ['DEBIT', 'DR']: return 'DEBIT'
        if t in ['CREDIT', 'CR']: return 'CREDIT'
        return 'UNKNOWN'

    def run(self):
        print("Doing hybrid search (Brute Force Cosine)...")
        # Compute full similarity matrix (N_bank x N_reg)
        sim_matrix = cosine_similarity(self.bank_vectors, self.reg_vectors)
        
        k = ANN_CANDIDATES
        all_candidates = []
        
        for i in range(len(self.bank)):
            row = self.bank.iloc[i]
            
            bank_amt = row['amount']
            bank_date = row['date']
            bank_type = self._normalize_type(row['type'])
            
            # Get Top-K indices from similarity matrix
            # argsort sorts ascending, so take last k and reverse
            top_k_indices = sim_matrix[i].argsort()[-k:][::-1]
            
            pool = set(top_k_indices)
            pool.update(self._get_amount_candidates(bank_amt))
            
            for reg_idx in pool:
                reg_row = self.reg.iloc[reg_idx]
                
                if self._normalize_type(reg_row['type']) != bank_type: continue
                
                date_diff = (bank_date - reg_row['date']).days
                if not (-DATE_LOOKBACK_DAYS <= date_diff <= DATE_LOOKAHEAD_DAYS): continue
                
                reg_amt = reg_row['amount']
                amt_diff = abs(bank_amt - reg_amt)
                max_amt = max(abs(bank_amt), abs(reg_amt))
                rel_diff = amt_diff / max_amt if max_amt > 0 else 0.0
                if rel_diff > MAX_AMOUNT_DIFF_PERCENT: continue
                
                text_score = sim_matrix[i, reg_idx]
                amount_score = 1.0 / (1.0 + rel_diff * 20)
                
                if date_diff < 0: date_score = 0.5 / (1.0 + abs(date_diff))
                else: date_score = 1.0 / (1.0 + abs(date_diff))
                
                final_score = 0.4*amount_score + 0.4*text_score + 0.2*date_score
                
                if final_score > MIN_CONFIDENCE:
                    all_candidates.append({
                        'bank_idx': i,
                        'reg_idx': reg_idx,
                        'score': final_score,
                        'bank_id': row['transaction_id'],
                        'match_reg_id': reg_row['transaction_id'],
                        'bank_true_id': row.get('true_id', -1),
                        'match_reg_true_id': reg_row.get('true_id', -1)
                    })
                    
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        matched_b = set()
        matched_r = set()
        final = []
        
        for c in all_candidates:
            if c['bank_idx'] not in matched_b and c['reg_idx'] not in matched_r:
                matched_b.add(c['bank_idx'])
                matched_r.add(c['reg_idx'])
                c['method'] = 'ML_EMBEDDING'
                final.append(c)
        
        return pd.DataFrame(final)
