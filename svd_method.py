import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from data_loader import load_data
from evaluation import evaluate_results
from config import (
    SVD_COMPONENTS, MIN_CONFIDENCE, MAX_AMOUNT_DIFF_PERCENT,
    DATE_LOOKBACK_DAYS, DATE_LOOKAHEAD_DAYS, ANN_CANDIDATES
)

class TermExtractor:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english', min_df=1)
        
    def fit(self, texts):
        self.vectorizer.fit(texts)
        
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

class AlignmentModel:
    def __init__(self):
        self.bank_vectorizer = CountVectorizer(stop_words='english', min_df=1)
        self.reg_vectorizer = CountVectorizer(stop_words='english', min_df=1)
        self.alignment_matrix = None

    def fit(self, bank_docs, reg_docs):
        print("Calculating Mutual Information (Alignment)...")
        self.bank_vectorizer.fit(bank_docs)
        self.reg_vectorizer.fit(reg_docs)
        
        B = self.bank_vectorizer.transform(bank_docs)
        R = self.reg_vectorizer.transform(reg_docs)
        
        C = B.T.dot(R)
        N = len(bank_docs)
        
        count_bank = np.array(B.sum(axis=0)).flatten()
        count_reg = np.array(R.sum(axis=0)).flatten()
        count_bank[count_bank == 0] = 1
        count_reg[count_reg == 0] = 1
        
        C_dense = C.toarray()
        denom = np.outer(count_bank, count_reg)
        ratio = (C_dense * N) / denom
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pmi = np.log(ratio)
        pmi[~np.isfinite(pmi)] = 0
        pmi[pmi < 0] = 0
        
        # Weighted PMI
        p_ab = C_dense / N
        self.alignment_matrix = p_ab * pmi

    def transform(self, bank_texts, reg_texts):
        bank_vec = self.bank_vectorizer.transform(bank_texts)
        reg_vec = self.reg_vectorizer.transform(reg_texts)
        aligned_bank = bank_vec.dot(self.alignment_matrix)
        return aligned_bank, reg_vec

class SVDReconciler:
    def __init__(self, n_components=SVD_COMPONENTS):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
    def fit(self, matrix):
        print(f"Fitting SVD ({self.svd.n_components} components)...")
        self.svd.fit(matrix)
        
    def transform(self, matrix):
        return self.svd.transform(matrix)

class SVDMatchingEngine:
    def __init__(self, bank_df, reg_df, bank_vectors, reg_vectors):
        self.bank = bank_df
        self.reg = reg_df
        self.bank_vectors = bank_vectors.astype('float32')
        self.reg_vectors = reg_vectors.astype('float32')
        
        # FAISS Index
        d = self.reg_vectors.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(self.bank_vectors)
        faiss.normalize_L2(self.reg_vectors)
        self.index.add(self.reg_vectors)
        
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
        print("Running SVD Matching...")
        k = ANN_CANDIDATES
        distances, indices = self.index.search(self.bank_vectors, k)
        
        all_candidates = []
        
        for i in range(len(self.bank)):
            row = self.bank.iloc[i]
            bank_amt = row['amount']
            bank_date = row['date']
            bank_type = self._normalize_type(row['type'])
            
            pool = set(indices[i])
            pool.update(self._get_amount_candidates(bank_amt))
            
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
                
                text_score = np.dot(self.bank_vectors[i], self.reg_vectors[reg_idx])
                amount_score = 1.0 / (1.0 + rel_diff * 20)
                
                if date_diff < 0: date_score = 0.5 / (1.0 + abs(date_diff))
                else: date_score = 1.0 / (1.0 + abs(date_diff))
                
                final_score = 0.5*amount_score + 0.3*text_score + 0.2*date_score
                
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
                final.append(c)
        
        return pd.DataFrame(final)

def get_unique_matches(bank_df, reg_df):
    print("Finding unique amount matches...")
    bc = bank_df['amount'].value_counts()
    rc = reg_df['amount'].value_counts()
    unique = bc[bc==1].index.intersection(rc[rc==1].index)
    
    pairs = []
    for amt in unique:
        b = bank_df[bank_df['amount']==amt].iloc[0]
        r = reg_df[reg_df['amount']==amt].iloc[0]
        if abs((b['date'] - r['date']).days) <= 5:
            pairs.append({'description_bank': b['description'], 'description_reg': r['description']})
    return pd.DataFrame(pairs)

def main():
    bank_df, reg_df = load_data()
    
    train_df = get_unique_matches(bank_df, reg_df)
    total_train = len(train_df)
    print(f"Total available training data: {total_train} pairs")
    
    # Run 3 Iterations with increasing training data size (33%, 66%, 100%)
    fractions = [0.33, 0.66, 1.0]
    
    for i, frac in enumerate(fractions):
        n_samples = int(total_train * frac)
        current_train = train_df.iloc[:n_samples]
        
        print(f"\n--- Iteration {i+1}: Training with {n_samples} pairs ({int(frac*100)}%) ---")
        
        alignment = AlignmentModel()
        alignment.fit(current_train['description_bank'], current_train['description_reg'])
        
        _, all_reg = alignment.transform([], reg_df['description'])
        
        svd = SVDReconciler(n_components=SVD_COMPONENTS)
        svd.fit(all_reg)
        
        aligned_bank, reg_vec = alignment.transform(bank_df['description'], reg_df['description'])
        bank_svd = svd.transform(aligned_bank)
        reg_svd = svd.transform(reg_vec)

        matcher = SVDMatchingEngine(bank_df, reg_df, bank_svd, reg_svd)
        matches = matcher.run()
        
        print(f"SVD Matcher found {len(matches)} matches")
        evaluate_results(matches, bank_df, reg_df)

if __name__ == "__main__":
    main()
