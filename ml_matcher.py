import numpy as np
import pandas as pd
import faiss
from config import (
    MIN_CONFIDENCE, MAX_AMOUNT_DIFF_PERCENT,
    DATE_LOOKBACK_DAYS, DATE_LOOKAHEAD_DAYS, ANN_CANDIDATES
)

class MLMatchingEngine:
    def __init__(self, bank_df, reg_df, bank_vectors, reg_vectors):
        self.bank = bank_df
        self.reg = reg_df
        self.bank_vectors = bank_vectors.astype('float32')
        self.reg_vectors = reg_vectors.astype('float32')
        
        d = self.reg_vectors.shape[1]
        self.index = faiss.IndexFlatIP(d)
        
        # Normalize for Cosine Similarity
        faiss.normalize_L2(self.bank_vectors)
        faiss.normalize_L2(self.reg_vectors)
        self.index.add(self.reg_vectors)
        
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
        print("Doing hybrid search (FAISS ANN)...")
        
        k = ANN_CANDIDATES
        distances, indices = self.index.search(self.bank_vectors, k)
        
        all_candidates = []
        
        for i in range(len(self.bank)):
            row = self.bank.iloc[i]
            
            bank_amt = row['amount']
            bank_date = row['date']
            bank_type = self._normalize_type(row['type'])
            
            text_candidate_indices = indices[i]
            text_candidate_scores = distances[i]
            
            text_score_map = {idx: score for idx, score in zip(text_candidate_indices, text_candidate_scores)}
            
            amount_candidate_indices = self._get_amount_candidates(bank_amt)
            
            pool = set(text_candidate_indices)
            pool.update(amount_candidate_indices)
            
            for reg_idx in pool:
                if reg_idx == -1: continue
                
                reg_row = self.reg.iloc[reg_idx]
                
                if self._normalize_type(reg_row['type']) != bank_type: continue
                
                date_diff = (bank_date - reg_row['date']).days
                if not (-DATE_LOOKBACK_DAYS <= date_diff <= DATE_LOOKAHEAD_DAYS): continue
                
                reg_amt = reg_row['amount']
                amt_diff = abs(bank_amt - reg_amt)
                max_amt = max(abs(bank_amt), abs(reg_amt))
                rel_diff = amt_diff / max_amt if max_amt > 0 else 0.0
                if rel_diff > MAX_AMOUNT_DIFF_PERCENT: continue
                
                if reg_idx in text_score_map:
                    text_score = text_score_map[reg_idx]
                else:
                    text_score = np.dot(self.bank_vectors[i], self.reg_vectors[reg_idx])
                
                amount_score = 1.0 / (1.0 + rel_diff * 20)
                
                if date_diff < 0: date_score = 0.5 / (1.0 + abs(date_diff))
                else: date_score = 1.0 / (1.0 + abs(date_diff))
                
                final_score = 0.5*amount_score + 0.4*text_score + 0.1*date_score
                
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
                c['method'] = 'ML_BERT'
                final.append(c)
        
        return pd.DataFrame(final)
