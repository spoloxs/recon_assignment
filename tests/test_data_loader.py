import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.data_loader import normalize_columns, load_data

class TestDataLoader(unittest.TestCase):

    def test_normalize_columns_valid(self):
        df = pd.DataFrame({
            'transaction_id': ['B1', 'B2'],
            'date': ['2023-01-01', '2023-01-02'],
            'amount': [100, 200]
        })
        normalized = normalize_columns(df, "Test")
        self.assertTrue('true_id' in normalized.columns)
        self.assertEqual(normalized['true_id'][0], 1)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(normalized['date']))

    def test_normalize_columns_missing_date(self):
        df = pd.DataFrame({'transaction_id': ['B1']})
        with self.assertRaises(ValueError):
            normalize_columns(df, "Test")

    @patch('src.common.data_loader.pd.read_csv')
    @patch('src.common.data_loader.os.path.exists')
    def test_load_data(self, mock_exists, mock_read_csv):
        mock_exists.return_value = True
        
        mock_bank = pd.DataFrame({'transaction_id': ['B1'], 'date': ['2023-01-01'], 'amount': [100]})
        mock_reg = pd.DataFrame({'transaction_id': ['R1'], 'date': ['2023-01-01'], 'amount': [100]})
        
        mock_read_csv.side_effect = [mock_bank, mock_reg]
        
        bank, reg = load_data()
        self.assertEqual(len(bank), 1)
        self.assertEqual(len(reg), 1)
        self.assertTrue('true_id' in bank.columns)

if __name__ == '__main__':
    unittest.main()
