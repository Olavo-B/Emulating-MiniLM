import unittest
import torch
from src.loss.distillation import DistillationLoss
from torch.nn import functional as F

# FILE: src/loss/test_distillation.py


class TestDistillationLoss(unittest.TestCase):
    def setUp(self):
        self.temperature = 2.0
        self.loss_fn = DistillationLoss(self.temperature)
    
    def test_forward(self):
        # Create sample input tensors
        batch_size = 1
        num_heads = 1
        seq_length = 2
        
        student_attention = torch.randn(batch_size, num_heads, seq_length, seq_length)
        teacher_attention = torch.randn(batch_size, num_heads, seq_length, seq_length)
        student_value_relation = torch.randn(batch_size, num_heads, seq_length, seq_length)
        teacher_value_relation = torch.randn(batch_size, num_heads, seq_length, seq_length)

        student_attention.requires_grad = True
        student_value_relation.requires_grad = True

        print(student_attention)
        print(teacher_attention)
        print(student_value_relation)
        print(teacher_value_relation)

        print(F.kl_div(teacher_attention, student_attention, reduction='none'))

        # Call the forward method
        loss = self.loss_fn.forward(student_attention, teacher_attention, 
                                    student_value_relation, teacher_value_relation)
        
        print(loss)

        # Assert the output type and value
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)

if __name__ == '__main__':
    unittest.main()