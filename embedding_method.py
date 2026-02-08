import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from data_loader import load_data
from config import (
    EMBEDDING_MODEL, MIN_CONFIDENCE, MAX_AMOUNT_DIFF_PERCENT,
    DATE_LOOKBACK_DAYS, DATE_LOOKAHEAD_DAYS, ANN_CANDIDATES
)
from evaluation import evaluate_results
class TransformerModel:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def fit(self, bank_texts, reg_texts, epochs=1):
        print("Fine-tuning started.")
        train_examples = []
        for b_text, r_text in zip(bank_texts, reg_texts):
            train_examples.append(InputExample(texts=[str(b_text), str(r_text)], label=1.0))
        # print(f"Created {len(train_examples)} training examples for fine-tuning.")
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        train_loss = losses.CosineSimilarityLoss(self.model)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs)
        print("Fine-tuning complete.")

    def transform(self, texts):
        return self.model.encode(texts.tolist(), show_progress_bar=False)

class MLMatchingEngine:
    def __init__(self, bank_df, reg_df, bank_vectors, reg_vectors):
        self.bank = bank_df
        self.reg = reg_df
        self.bank_vectors = bank_vectors
        self.reg_vectors = reg_vectors
        
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
        print("Doing hybrid search...")
        sim_matrix = cosine_similarity(self.bank_vectors, self.reg_vectors)
        
        k = ANN_CANDIDATES
        all_candidates = []
        
        for i in range(len(self.bank)):
            row = self.bank.iloc[i]
            
            bank_amt = row['amount']
            bank_date = row['date']
            bank_type = self._normalize_type(row['type'])
            
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
                c['method'] = 'ML_BERT'
                final.append(c)
        
        return pd.DataFrame(final)

if __name__ == "__main__":
    bank_df , reg_df = load_data()

    # Before fine tuning
    print("Running ML Matcher before fine-tuning...")
    transformer = TransformerModel()
    bank_vecs = transformer.transform(bank_df['description'])
    reg_vecs = transformer.transform(reg_df['description'])
    ml_matcher = MLMatchingEngine(bank_df, reg_df, bank_vecs, reg_vecs)
    ml_matches = ml_matcher.run()
    print(f"ML Matcher found {len(ml_matches)} matches")
    print(evaluate_results(ml_matches, bank_df, reg_df))

    # After fine tuning
    print("\nFine-tuning the model with transaction descriptions...")
    transformer.fit(bank_df['description'], reg_df['description'], epochs=100)
    bank_vecs = transformer.transform(bank_df['description'])
    reg_vecs = transformer.transform(reg_df['description'])
    ml_matcher = MLMatchingEngine(bank_df, reg_df, bank_vecs, reg_vecs)
    ml_matches = ml_matcher.run()
    print(f"ML Matcher found {len(ml_matches)} matches")
    print(evaluate_results(ml_matches, bank_df, reg_df))
