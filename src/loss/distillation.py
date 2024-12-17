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
        attention_dist = F.softmax(attention_scores / self.temperature, dim=-1) + 1e-8
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
            theacher_A, teacher_values,
            student_A, student_values):
        """
        Computes the distillation loss between teacher and student models 
        with proper LAT (attention transfer) and VR (value relation) scaling.

        Args:
            theacher_A (torch.Tensor): Attention distribution from the teacher model
            teacher_values (torch.Tensor): Value vectors from the teacher model
            student_A (torch.Tensor): Attention distribution from the student model
            student_values (torch.Tensor): Value vectors from the student model
        
        Returns:
            torch.Tensor: Total distillation loss with LAT and VR scaling.
        """
        # Ensure input tensors require gradients
        student_A.requires_grad_(True)
        theacher_A.requires_grad_(True)
        student_values.requires_grad_(True)
        teacher_values.requires_grad_(True)


        # Attention Distribution Transfer (LAT)
        student_A = self._compute_attention_distribution(student_A)
        theacher_A = self._compute_attention_distribution(theacher_A)
        
        kl_attention = F.kl_div(
            torch.log(student_A), theacher_A, reduction="none"
        )  # Shape: [batch, A_h, |x|, |x|]

        Ah = student_A.size(1)  # Number of attention heads
        X = student_A.size(2)   # Sequence length

        attention_transfer_loss = kl_attention.sum(dim=(-2, -1)).mean() / (Ah * X)

        # Value Relation Transfer (VR)
        teacher_value_relation = self._compute_value_relation(teacher_values)  # Shape: [batch, A_h, |x|, |x|]
        student_value_relation = self._compute_value_relation(student_values)  # Shape: [batch, A_h, |x|, |x|]

        kl_value_relation = F.kl_div(
            torch.log(student_value_relation+ 1e-8), 
            teacher_value_relation, 
            reduction="none"
        )  # Shape: [batch, A_h, |x|, |x|]

        value_relation_loss = kl_value_relation.sum(dim=(-2, -1)).mean() / (Ah * X)

        # Total loss
        total_loss = attention_transfer_loss + value_relation_loss

        return total_loss
