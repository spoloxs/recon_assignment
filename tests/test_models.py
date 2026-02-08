import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embedding.model import TransformerModel

class TestModels(unittest.TestCase):

    @patch('src.embedding.model.SentenceTransformer')
    def test_transform(self, mock_transformer):
        # Mock encode
        mock_instance = mock_transformer.return_value
        mock_instance.encode.return_value = [[0.1, 0.2]]
        
        model = TransformerModel()
        result = model.transform(["text"])
        
        mock_instance.encode.assert_called_once()
        self.assertEqual(result, [[0.1, 0.2]])

    @patch('src.embedding.model.SentenceTransformer')
    @patch('src.embedding.model.DataLoader')
    @patch('src.embedding.model.losses.CosineSimilarityLoss')
    def test_fit(self, mock_loss, mock_dataloader, mock_transformer):
        mock_instance = mock_transformer.return_value
        # Mock fit
        
        model = TransformerModel()
        model.fit(["B1"], ["R1"], epochs=1)
        
        # Verify calls
        mock_instance.fit.assert_called()

if __name__ == '__main__':
    unittest.main()
