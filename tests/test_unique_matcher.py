import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.unique_matcher import UniqueAmountMatcher

class TestUniqueMatcher(unittest.TestCase):

    def setUp(self):
        self.bank_df = pd.DataFrame({
            'transaction_id': ['B1', 'B2', 'B3'],
            'amount': [100.0, 50.0, 50.0],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
            'true_id': [1, 2, 3]
        })
        self.reg_df = pd.DataFrame({
            'transaction_id': ['R1', 'R2', 'R3'],
            'amount': [100.0, 50.0, 50.0],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-05']),
            'true_id': [1, 2, 3]
        })

    def test_find_matches(self):
        matcher = UniqueAmountMatcher(self.bank_df, self.reg_df)
        matches, rem_b, rem_r = matcher.find_matches()
        
        # 100.0 is unique -> Match
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.iloc[0]['bank_id'], 'B1')
        self.assertEqual(matches.iloc[0]['match_reg_id'], 'R1')
        
        # 50.0 is not unique -> Remaining
        self.assertEqual(len(rem_b), 2)
        self.assertEqual(len(rem_r), 2)
        self.assertTrue('B2' in rem_b['transaction_id'].values)

if __name__ == '__main__':
    unittest.main()
