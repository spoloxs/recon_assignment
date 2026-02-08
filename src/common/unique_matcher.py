import pandas as pd
from .data_loader import load_data
from .config import UNIQUE_DATE_TOLERANCE
class UniqueAmountMatcher:
    def __init__(self, bank_df, reg_df):
        self.bank = bank_df
        self.reg = reg_df
        
    def find_matches(self):
        """
        Identifies transaction pairs where amount is unique in both datasets.
        Returns: 
            - matches_df: The matched pairs.
            - remaining_bank: Unmatched bank transactions.
            - remaining_reg: Unmatched register transactions.
        """
        print("Starting Unique Amount Matching...")
        
        bank_counts = self.bank['amount'].value_counts()
        reg_counts = self.reg['amount'].value_counts()
        
        unique_bank_amts = bank_counts[bank_counts == 1].index
        unique_reg_amts = reg_counts[reg_counts == 1].index
        
        common_amounts = unique_bank_amts.intersection(unique_reg_amts)
        print(f"Found {len(common_amounts)} potential unique matches")
        
        matches = []
        matched_bank_indices = set()
        matched_reg_indices = set()
        
        for amt in common_amounts:
            b_idx = self.bank[self.bank['amount'] == amt].index[0]
            r_idx = self.reg[self.reg['amount'] == amt].index[0]
            
            b_row = self.bank.loc[b_idx]
            r_row = self.reg.loc[r_idx]
            
            date_diff = (b_row['date'] - r_row['date']).days
            
            flag = None
            if abs(date_diff) > 0:
                flag = f"Date diff: {date_diff}"
            
            matches.append({
                'bank_idx': b_idx,
                'reg_idx': r_idx,
                'bank_id': b_row['transaction_id'],
                'match_reg_id': r_row['transaction_id'],
                'amount': amt,
                'score': 1.0,
                'method': 'UniqueAmount',
                'flag': flag,
                'bank_true_id': b_row.get('true_id', -1),
                'match_reg_true_id': r_row.get('true_id', -1)
            })
            
            matched_bank_indices.add(b_idx)
            matched_reg_indices.add(r_idx)
            
        matches_df = pd.DataFrame(matches)
        
        remaining_bank = self.bank.drop(matched_bank_indices)
        remaining_reg = self.reg.drop(matched_reg_indices)
        
        print(f"Matched {len(matches_df)} transactions via Unique Amount.")
        return matches_df, remaining_bank, remaining_reg

if __name__ == "__main__":
    bank_df , reg_df = load_data()
    
    matcher = UniqueAmountMatcher(bank_df, reg_df)
    matches, rem_bank, rem_reg = matcher.find_matches()
    
    print(matches.head())
    print(f"Remaining Bank Transactions: {len(rem_bank)}")
    print(f"Remaining Register Transactions: {len(rem_reg)}")