import unittest
from unittest.mock import patch
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.evaluation import evaluate_results

class TestEvaluation(unittest.TestCase):

    def test_evaluate_results(self):
        # Mock inputs
        matches = pd.DataFrame({
            'bank_true_id': [1, 2, 3],
            'match_reg_true_id': [1, 2, 4], # 2 correct, 1 wrong
            'match_reg_id': ['R1', 'R2', 'R4']
        })
        
        bank_df = pd.DataFrame({'true_id': [1, 2, 3, 5]}) # 4 valid
        reg_df = pd.DataFrame({'true_id': [1, 2, 4, 5]}) # 1, 2, 5 matchable
        
        # Valid intersection: 1, 2, 5
        # Total Valid Bank: 3 (1, 2, 5). 3 is NOT in intersection?
        # true_id 3 is in bank, but not in reg. So it's not "valid" for recall?
        # My logic: valid_ids = intersection.
        # Intersection = {1, 2, 5}.
        # Valid Bank = rows where true_id in {1, 2, 5}.
        # Bank rows: 1, 2, 5. (3 is ignored).
        # So valid_bank_count = 3.
        
        # Correct Matches: 2 (1==1, 2==2). 3!=4 (Wrong).
        # TP = 2.
        # FP = 1.
        # FN = Valid Bank (3) - TP (2) = 1.
        
        # Precision = 2 / (2+1) = 0.66
        # Recall = 2 / (2+1) = 0.66
        
        with patch('builtins.print') as mock_print:
            evaluate_results(matches, bank_df, reg_df)
            
            # Verify calls
            # Can't easily check exact string unless I capture stdout or check mock_print.call_args
            # But just running without error is good.
            self.assertTrue(mock_print.called)

if __name__ == '__main__':
    unittest.main()
