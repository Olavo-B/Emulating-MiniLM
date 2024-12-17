import unittest
import torch
from src.model.BERT import BERT

# FILE: src/model/test_BERT.py


class TestBERT(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 30522
        self.d_model = 512
        self.n_head = 8
        self.n_layers = 6
        self.d_ff = 2048
        self.max_len = 5000
        self.dropout = 0.1
        self.model = BERT(self.vocab_size, self.d_model, self.n_head, self.n_layers, 
                          self.d_ff, self.max_len, self.dropout)
    
    def test_forward(self):
        # Create a sample input tensor
        input_ids = torch.randint(0, self.vocab_size, (1, 128))  # Batch size of 1, sequence length of 128
        attention_mask = torch.ones((1, 128))  # Attention mask

        # Call the forward method
        logits = self.model.forward(input_ids, attention_mask)

        # Assert the output shape and type
        self.assertEqual(logits.shape, (1, 128, self.d_model)) # Output shape should be (batch_size, seq_length,
        self.assertIsInstance(logits, torch.Tensor)

if __name__ == '__main__':
    # Run the unit test, if everything is fine, it will print OK
    
    unittest.main()