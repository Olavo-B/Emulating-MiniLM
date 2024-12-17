'''
/***********************************************
 * File: BERT.py
 * Author: Olavo Alves Barros Silva
 * Contact: olavo.barros@ufv.com
 * Date: 2024-12-16
 * License: [License Type]
 * Description: Enhanced BERT model with classification support and attention feature extraction
 ***********************************************/
'''

import torch
import torch.nn as nn
from sklearn.utils import Bunch
import torch.nn.functional as F

from src.model.myTrasformers import EncoderLayer, PositionalEncoding

class BERT(nn.Module):
    '''
    Enhanced BERT model with classification head and attention feature extraction.

    Args:
        vocab_size (int): The size of the vocabulary.
        num_labels (int): Number of classification labels.
        d_model (int): The dimension of the model.
        n_head (int): The number of heads.
        n_layers (int): The number of layers.
        d_ff (int): The dimension of the feedforward layer.
        max_len (int): The maximum length of the input.
        dropout (float): The dropout rate.
    '''
    def __init__(self, 
                 vocab_size, 
                 num_labels=2, 
                 d_model=768, 
                 n_head=12, 
                 n_layers=6, 
                 d_ff=2048, 
                 max_len=5000, 
                 dropout=0.1):
        super(BERT, self).__init__()

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout, max_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)
        ])

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels)
        )

        # Stores intermediate results for feature extraction
        self.intermediate_outputs = []

    def forward(self, x, attention_mask=None, 
                output_attentions=False, return_dict = False):
        '''
        Forward pass with optional classification.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The attention mask tensor.
        
        Returns:
            torch.Tensor: Logits for classification
        '''
        # Reset intermediate outputs
        self.intermediate_outputs = []

        # Initial embedding
        x = self.embedding(x)
        x = self.pe(x)

        # Store initial embedding
        self.intermediate_outputs.append(x)


        # Process through transformer layers
        hidden_states = []
        attentions    = []
        for layer in self.layers:
            if attention_mask is not None:
                x = layer(x, attention_mask, output_attentions, return_dict)
            else:
                x = layer(x, output_attentions, return_dict)
            
            if return_dict:
                hidden_states.append(x.hidden_states)
                attentions.append(x.attentions)

            self.intermediate_outputs.append(x)

        # Use the last layer's output for classification
        # Take the [CLS] token (first token) for classification
        if return_dict:
            cls_representation = x.hidden_states[:, 0, :]
        else:
            cls_representation = x[:, 0, :]
        logits = self.classification_head(cls_representation)

        if return_dict:
            return Bunch(
                hidden_states=hidden_states,
                attentions=attentions
            )
        
        return logits

    def extract_attention_features(self, x, attention_mask=None):
        '''
        Extract attention-related features for knowledge distillation.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The attention mask tensor.
        
        Returns:
            dict: Dictionary containing queries, keys, values, and other features
        '''
        # Reset intermediate outputs
        self.intermediate_outputs = []

        # Initial embedding
        x = self.embedding(x)
        x = self.pe(x)

        # Store initial embedding
        self.intermediate_outputs.append(x)


        # Process through transformer layers
        hidden_states = []
        attentions    = []
        for i,layer in enumerate(self.layers):
            x = x if i == 0 else x.hidden_states
            if attention_mask is not None:
                x = layer(x, attention_mask, 
                          output_attentions=True, return_dict=True)
            else:
                x = layer(x, output_attentions=True, return_dict=True)
            
            hidden_states.append(x.hidden_states)
            attentions.append(x.attention)

            self.intermediate_outputs.append(x)

        # Use the last layer's output for classification
        # Take the [CLS] token (first token) for classification
        cls_representation = x.hidden_states[:, 0, :]
        logits = self.classification_head(cls_representation)


        return Bunch(
            hidden_states=hidden_states,
            attentions=attentions
        )
    
    def get_intermediate_outputs(self):
        '''
        Retrieve intermediate layer outputs.

        Returns:
            list: List of intermediate outputs from embedding and transformer layers
        '''
        return self.intermediate_outputs