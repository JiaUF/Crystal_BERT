from .Transformer import Transformer
from .Embedding import Total_Embed
import torch.nn as nn
import numpy as np

class BERT(nn.Module):
    "BERT Model"
    def __init__(self, heads=12, d_model=120, hidden=716, dropout=0.1, n_layers=10):
        """
        params:
            heads: number of attention heads
            d_model: dimension of embed
            hidden: dimension of hidden model size
            dropout: dropout rate
            n_layers: number of transformer block
        """
        super(BERT, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.n_layers = n_layers
        
        #feed_forward hidden is conventionally 4*hidden
        self.d_ff = 4*hidden
        
        #embedding
        self.embedding = Total_Embed(d_model = d_model, dropout = dropout)
        
        #number of transformer block
        self.transformer_blocks = nn.ModuleList([Transformer(d_model=d_model, d_ff=4*hidden, heads = heads, dropout = dropout) for _ in range(n_layers)])
            
    def forward(self, bert_input, mask):
        #embed input
        x = self.embedding(bert_input)
        mask = mask.unsqueeze(1).repeat(1, bert_input.size(1), 1).unsqueeze(1)
        #run over the number of transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, x, x, mask)
        
        return x