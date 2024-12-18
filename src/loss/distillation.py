'''
/***********************************************
 * File: distillation.py
 * Author: Olavo Alves Barros Silva
 * Contact: olavo.barros@ufv.br
 * Date: 2024-12-02
 * License: [License Type]
 * Description: This file contains the implementation of the distillation loss.
 * This loss is proposed by Wang et al. (2020).
 ***********************************************/
 '''

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Implements the distillation loss for MINILM as described in the paper.
    
    This loss consists of two components:
    1. Attention Distribution Transfer Loss
    2. Value-Relation Transfer Loss
    
    The loss captures the knowledge transfer between teacher and student models
    by minimizing the KL-divergence of attention distributions and value relations.
    """

    def __init__(self, temperature=2.0):
        """
        Initializes the DistillationLoss module.

        Args:
            temperature (float): Softening temperature for the softmax distributions.
                               Default is 2.0 as suggested in the paper.
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature

    def _compute_attention_distribution(self, attention_scores):
        """
        Computes the attention distribution using scaled dot-product.
        
        Args:
            queries (torch.Tensor): Query vectors
            keys (torch.Tensor): Key vectors
        
        Returns:
            torch.Tensor: Attention distribution
        """
 
        
        # Apply softmax with temperature scaling
        attention_dist = F.softmax(attention_scores / self.temperature, dim=-1)
        return attention_dist

    def _compute_value_relation(self, values):
        """
        Computes the value relation matrix using scaled dot-product.
        
        Args:
            values (torch.Tensor): Value vectors
        
        Returns:
            torch.Tensor: Value relation matrix
        """
        # Compute scaled dot-product between values
        value_relation = torch.matmul(values, values.transpose(-2, -1))
        value_relation = value_relation / (values.size(-1) ** 0.5)
        
        # Apply softmax
        value_relation_dist = F.softmax(value_relation, dim=-1)
        return value_relation_dist

    def forward(self, 
            teacher_A, teacher_values,
            student_A, student_values):
        """
        Computes the distillation loss between teacher and student models 
        with proper LAT (attention transfer) and VR (value relation) scaling.

        Args:
            teacher_A (torch.Tensor): Attention distribution from the teacher model
            teacher_values (torch.Tensor): Value vectors from the teacher model
            student_A (torch.Tensor): Attention distribution from the student model
            student_values (torch.Tensor): Value vectors from the student model
        
        Returns:
            torch.Tensor: Total distillation loss with LAT and VR scaling.
        """

        # Ensure all tensors require gradients
        # Check gradients of relevant tensors

        # Ensure all tensors involved in loss computation are part of the computation graph


        student_A = self._compute_attention_distribution(student_A)
        teacher_A = self._compute_attention_distribution(teacher_A)

        
        # Attention Distribution Transfer (LAT) - KL Divergence
        kl_attention = F.kl_div(
            torch.log(student_A + 1e-8),  # Log with stability added
            teacher_A, reduction="none"
        )  # Shape: [batch, A_h, |x|, |x|]

        Ah = student_A.size(1)  # Number of attention heads
        X = student_A.size(2)   # Sequence length

        # Summing over sequence positions and averaging over attention heads
        attention_transfer_loss = kl_attention.sum(dim=(-2, -1)).mean() / (Ah * X)

        # Value Relation Transfer (VR) - KL Divergence for value relations
        teacher_value_relation = self._compute_value_relation(teacher_values)
        student_value_relation = self._compute_value_relation(student_values)

        kl_value_relation = F.kl_div(
            torch.log(student_value_relation + 1e-8),  # Log with stability added
            teacher_value_relation, reduction="none"
        )  # Shape: [batch, A_h, |x|, |x|]

        # Summing over sequence positions and averaging over attention heads
        value_relation_loss = kl_value_relation.sum(dim=(-2, -1)).mean() / (Ah * X)

        # Total loss as the sum of both components
        total_loss = attention_transfer_loss + value_relation_loss

        # Ensure total loss requires gradients
        assert total_loss.requires_grad, "Total loss must require gradients!"

        return total_loss
