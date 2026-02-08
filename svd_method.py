import pandas as pd
from src.common.data_loader import load_data
from src.common.evaluation import evaluate_results
from src.common.preprocessing import TermExtractor
from src.common.alignment import AlignmentModel
from src.svd.model import SVDReconciler
from src.svd.matcher import SVDMatchingEngine
from src.common.config import SVD_COMPONENTS

def get_unique_matches(bank_df, reg_df):
    print("Finding unique amount matches...")
    bc = bank_df['amount'].value_counts()
    rc = reg_df['amount'].value_counts()
    unique = bc[bc==1].index.intersection(rc[rc==1].index)
    
    pairs = []
    for amt in unique:
        b = bank_df[bank_df['amount']==amt].iloc[0]
        r = reg_df[reg_df['amount']==amt].iloc[0]
        if abs((b['date'] - r['date']).days) <= 10:
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
