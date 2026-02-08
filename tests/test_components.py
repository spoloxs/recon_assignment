import pytest
import pandas as pd
import sys
import os

# Add parent directory to path so we can import from main/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unique_matcher import UniqueAmountMatcher
from data_loader import normalize_columns

class TestComponents:
    def test_normalize_columns(self):
        # Create dummy data
        data = {
            'transaction_id': ['B001', 'B002'],
            'date': ['2023-01-01', '2023-01-02'],
            'amount': [100.0, 200.0]
        }
        df = pd.DataFrame(data)
        
        # Run normalization
        normalized_df = normalize_columns(df, "Test")
        
        # Check results
        assert 'true_id' in normalized_df.columns
        assert normalized_df['true_id'].iloc[0] == 1
        assert pd.api.types.is_datetime64_any_dtype(normalized_df['date'])

    def test_unique_matcher(self):
        # Setup mock bank data
        bank_data = {
            'transaction_id': ['B1', 'B2', 'B3'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
            'description': ['Unique', 'Common', 'Common'],
            'amount': [100.0, 50.0, 50.0],
            'type': ['DEBIT', 'DEBIT', 'DEBIT']
        }
        bank_df = pd.DataFrame(bank_data)
        bank_df['true_id'] = [1, 2, 3] # Pre-populate true_id as normalize does
        
        # Setup mock register data
        reg_data = {
            'transaction_id': ['R1', 'R2', 'R3'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-05']),
            'description': ['Unique', 'Common 1', 'Common 2'],
            'amount': [100.0, 50.0, 50.0],
            'type': ['DR', 'DR', 'DR']
        }
        reg_df = pd.DataFrame(reg_data)
        reg_df['true_id'] = [1, 2, 3]

        # Run Matcher
        matcher = UniqueAmountMatcher(bank_df, reg_df)
        matches, rem_b, rem_r = matcher.find_matches()
        
        # Assertions
        # 100.0 is unique -> Should match
        assert len(matches) == 1
        assert matches.iloc[0]['amount'] == 100.0
        assert matches.iloc[0]['bank_id'] == 'B1'
        assert matches.iloc[0]['match_reg_id'] == 'R1'
        
        # 50.0 is not unique -> Should not match
        assert len(rem_b) == 2
        assert len(rem_r) == 2
