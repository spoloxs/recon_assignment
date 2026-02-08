import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_matcher import MLMatchingEngine

class TestMLMatcher(unittest.TestCase):

    def test_run(self):
        bank_df = pd.DataFrame({
            'transaction_id': ['B1', 'B2'],
            'amount': [100.0, 50.0],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'type': ['DEBIT', 'DEBIT'],
            'true_id': [1, 2]
        })
        reg_df = pd.DataFrame({
            'transaction_id': ['R1', 'R2'],
            'amount': [100.0, 50.0],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'type': ['DR', 'DR'],
            'true_id': [1, 2]
        })
        
        # Perfect matching vectors
        bank_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        reg_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        engine = MLMatchingEngine(bank_df, reg_df, bank_vecs, reg_vecs)
        matches = engine.run()
        
        # Should match both
        self.assertEqual(len(matches), 2)
        # B1 -> R1
        match_b1 = matches[matches['bank_id'] == 'B1'].iloc[0]
        self.assertEqual(match_b1['match_reg_id'], 'R1')
        
        # B2 -> R2
        match_b2 = matches[matches['bank_id'] == 'B2'].iloc[0]
        self.assertEqual(match_b2['match_reg_id'], 'R2')

if __name__ == '__main__':
    unittest.main()
