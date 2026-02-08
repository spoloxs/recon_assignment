import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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
        
        p_ab = C_dense / N
        self.alignment_matrix = p_ab * pmi

    def transform(self, bank_texts, reg_texts):
        bank_vec = self.bank_vectorizer.transform(bank_texts)
        reg_vec = self.reg_vectorizer.transform(reg_texts)
        aligned_bank = bank_vec.dot(self.alignment_matrix)
        return aligned_bank, reg_vec
