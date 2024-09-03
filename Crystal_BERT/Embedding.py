import torch
import torch.nn.functional as Func
import torch.nn as nn
import numpy as np

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class One_Hot_Embed(nn.Module):
    "One_Hot Encoding 120 Vector"
    def __init__(self, d_model = 120):
        super(One_Hot_Embed, self).__init__()
        self.d_model = d_model
        self.register_buffer("hot_vec", None)
        self.linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        hot_vec = None
        hot_vec = Func.one_hot(x[:, :, 0].type(torch.int64), num_classes = self.d_model)
        hot_vec = hot_vec.to(torch.float32)
        emb_hot_vec = self.linear(hot_vec)
        return emb_hot_vec
    
class Embed3D(nn.Module):
    def __init__(self, c = 10000, d_model = 120):
        super(Embed3D, self).__init__()
        self.c = c
        self.d_model = 120
        D = int(np.ceil(d_model/6) * 2)
        self.div_term = 1/(c**(torch.arange(0, D, 2).float()/self.d_model))
        self.register_buffer("pe", None)
        
    def forward(self, x):
        """
        input: tensor (Batch, len, 4)
        
        output: positional encoding (Batch, len, d_model)
        """
        if self.pe is not None and self.pe == x.shape:
            return self.cached_penc
        self.pe = None
        N, L, r = x.shape
        emb = torch.zeros(N*L, self.d_model)
        x_pos = x[:, :, 1:4][:, :, 0].flatten()
        y_pos = x[:, :, 1:4][:, :, 1].flatten()
        z_pos = x[:, :, 1:4][:, :, 2].flatten()
        for i in range(N*L):
            k, l, m = x_pos[i], y_pos[i], z_pos[i]
            z = torch.stack((get_emb(self.div_term*k), get_emb(self.div_term*l), get_emb(self.div_term*m))).flatten()
            emb[i] = z
        self.pe = torch.reshape(emb, (N,L,self.d_model))
        return self.pe
    
class Total_Embed(nn.Module):
    """
    total embed: one_hot projection + positional embedding
    """
    def __init__(self, d_model=120, c=10000, dropout=0.1):
        super(Total_Embed, self).__init__()
        self.positional = Embed3D(c = c, d_model = d_model)
        self.One_Hot_Embed = One_Hot_Embed(d_model=d_model)
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, bert_input):
        x = self.positional(bert_input) + self.One_Hot_Embed(bert_input)
        return self.dropout(x)