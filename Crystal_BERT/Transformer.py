import torch 
import torch.nn as nn
import math 
from .Embedding import Total_Embed
import numpy as np
import torch.nn.functional as Func

class MultiHeadedAttention(nn.Module):
    """
    Computes the multiheaded attention
    """
    def __init__(self, heads, d_model):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model
        
        self.queries = nn.Linear(self.d_model, self.d_model, bias=False)
        self.keys = nn.Linear(self.d_model, self.d_model, bias=False)
        self.values = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.fc_output = nn.Linear(self.d_model, self.d_model)
        
    def forward(self, queries, keys, values, mask=None):
        N = queries.shape[0]
        query_len, key_len, value_len = queries.shape[1], keys.shape[1], values.shape[1]
        
        #reshaping query, keys, values
        #(N, max_len, d_model) -> (N, max_len, heads, d_k)
        queries = queries.reshape(N, query_len, self.heads, self.d_k)
        keys = keys.reshape(N, key_len, self.heads, self.d_k)
        values = values.reshape(N, value_len, self.heads, self.d_k)
        
        #compute the energy = q*k.T
        #[(N, queries_len, heads, d_k), (N, keys_len, heads, d_k)] -> (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys])
        
        #apply masking to the attention: set padded tokens to -inf so it does not effect attention score when softmax is applied
        if mask is not None:
            energy = energy.masked_fill(mask == 0, value = -float("inf"))
        
        #compute attention weights by appling softmax to scaled energy by (dim)^1/2
        attn_W = Func.softmax(energy / math.sqrt(self.d_model), dim=3)
        #compute the attention (attn*V)
        #(N, heads, query_len, key_len)*(N, value_len, heads, d_k) -> (N, query_len, heads, d_k)
        #reshape to (N, len, d_model)
        attention = torch.einsum("nhql, nlhd -> nqhd", [attn_W, values]).reshape(N, query_len, self.d_model)
        
        #send to the last linear layer
        #out: (N, len, d_model)
        out = self.fc_output(attention)
        return out
        
class FeedFoward(nn.Module):
    """
    Implements feedforward with GELU activation
    """
    def __init__(self, d_model, d_ff, dropout):
        super(FeedFoward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = Func.gelu
        
    def forward(self, x):
        #Feedforward layer
        #(N, len, d_model) -> (N, len, d_ff) -> (N, len, d_model)
        return self.W_2(self.dropout(self.activation(self.W_1(x))))
    

class Transformer(nn.Module):
    """
    Transformer block
    """
    def __init__(self, d_model, d_ff, heads, dropout):
        super(Transformer, self).__init__()
        #Initialize layers
        self.Attention = MultiHeadedAttention(heads, d_model)
        self.FeedFoward = FeedFoward(d_model, d_ff, dropout)
        self.Dropout = nn.Dropout(dropout)
        
        #layer norms
        self.Norm_1 = nn.LayerNorm(d_model)
        self.Norm_2 = nn.LayerNorm(d_model)
        
    def forward(self, queries, keys, values, mask):
        #queries, keys, values are all of the same dimension (N, len, d_model)
        attention = self.Attention(queries, keys, values, mask)
        
        #after multi-headed attention, do layer norm
        x = self.Dropout(self.Norm_1(attention + queries))
        #forward layer
        forward = self.FeedFoward(x)
        #skipped connection and layer norm
        out = self.Dropout(self.Norm_2(x+forward))
        return out