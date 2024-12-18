'''
/***********************************************
 * File: trasformers.py
 * Author: Olavo Alves Barros Silva
 * Contact: olavo.barros@ufv.com
 * Date: 2024-12-01
 * License: [License Type]
 * Description: This file contais the implementation of the transformers models.
 ***********************************************/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import Bunch
from torch.autograd import Variable

import torchvision
import torchvision.models as models

import math

class PositionalEncoding(nn.Module):
    '''
    This class implements the Positional Encoding for the transformers model.

    Args:
        d_model (int): The dimension of the model.
        dropout (float): The dropout rate.
        max_len (int): The maximum length of the input.
    
    Returns:
        x (torch.Tensor): The input tensor with the positional encoding.
    '''
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    '''
    This class implements the Multi-Head Attention for the transformers model.

    Args:
        d_model (int): The dimension of the model.
        n_head (int): The number of heads.
        dropout (float): The dropout rate.
    
    Returns:
        x (torch.Tensor): The input tensor with the multi-head attention.
    '''
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_length = q.size(1)

        # Linear projections
        q = self.w_q(q).view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)

        # Apply attention
        x, attention = self.attention(q, k, v, mask)  # [batch_size, n_head, seq_len, d_k]

        # Reshape back to [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.n_head * self.d_k)

        # Final linear transformation
        return self.w_o(x), attention

    
    def attention(self, q, k, v, mask=None):
        """
        Scaled dot-product attention.
        Args:
            q, k, v (torch.Tensor): Query, key, and value tensors.
            mask (torch.Tensor): Attention mask for padding (optional).
            
        Returns:
            torch.Tensor: The weighted sum of values after attention.
        """
        # Calculate scaled dot-product scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) 
        scores = attention_scores / math.sqrt(self.d_k)  # Scale by square root of d_k
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Mask out padding

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, n_head, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)

        # Multiply weights with values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, n_head, seq_len, d_k]

        return attn_output, scores

    
class FeedForward(nn.Module):
    '''
    This class implements the Feed Forward for the transformers model.

    Args:
        d_model (int): The dimension of the model.
        d_ff (int): The dimension of the feed forward network.
        dropout (float): The dropout rate.
    
    Returns:
        x (torch.Tensor): The input tensor with the feed forward.
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    '''
    This class implements the Encoder Layer for the transformers model.
    
    Args:
        d_model (int): The dimension of the model.
        n_head (int): The number of heads.
        d_ff (int): The dimension of the feed forward network.
        dropout (float): The dropout rate.
    
    Returns:
        x (torch.Tensor): The input tensor with the encoder layer.
    '''

    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
    
    def forward(self, x, mask, output_attentions=False, return_dict = False):

        output, attention = self.attn(x, x, x, mask)
        
        x = self.norm_1(x + self.dropout_1(output))
        x = self.norm_2(x + self.dropout_2(self.ff(x)))

        if output_attentions and return_dict:
            return Bunch(hidden_states=x, 
                         attention=attention)
        return x

# Manually compute Q and K using the attention weights
class ExtractQK(torch.nn.Module):
    def __init__(self, teacher_model):
        super().__init__()
        self.query_proj = teacher_model.bert.encoder.layer[-1].attention.self.query  # Extract Q projection matrix
        self.key_proj = teacher_model.bert.encoder.layer[-1].attention.self.key      # Extract K projection matrix

    def forward(self, hidden_states):
        # Compute Q and K
        Q = self.query_proj(hidden_states)  # Shape: [batch_size, seq_len, d_model]
        K = self.key_proj(hidden_states)    # Shape: [batch_size, seq_len, d_model]
        
        # Reshape Q and K for multi-head attention
        batch_size, seq_len, d_model = Q.size()
        n_head = 12  # Number of heads in BERT-base
        head_dim = d_model // n_head

        Q = Q.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)  # [batch_size, n_head, seq_len, head_dim]
        K = K.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)  # [batch_size, n_head, seq_len, head_dim]

        # Compute QK^T
        QK_T = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, n_head, seq_len, seq_len]
        return QK_T