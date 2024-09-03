import torch.nn as nn
from .BERT import BERT

class Atomic_LM(nn.Module):
    """
    Masked Language Model for Predicting Atomic Number"
    """
    def __init__(self, d_model):
        super(Atomic_LM, self).__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.atom = nn.Linear(d_model, 120)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        return self.softmax(self.atom(self.linear(x)))
    
class Position_LM(nn.Module):
    """
    Masked Language for Predicting Position
    """
    def __init__(self, d_model):
        super(Position_LM, self).__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.position = nn.Linear(d_model, 3)
        self.activation = nn.ReLU()
    def forward(self, x):
        return self.position(self.activation(self.linear(x)))
    
class Crystal_BERT(nn.Module):
    """
    Crystal BERT model to predict both position and atomic number
    """
    def __init__(self, bert : BERT):
        super(Crystal_BERT, self).__init__()
        self.bert = bert
        self.atom = Atomic_LM(self.bert.d_model)
        self.position = Position_LM(self.bert.d_model)
        
    def forward(self, bert_input, mask):
        x = self.bert(bert_input, mask)
        return self.atom(x), self.position(x)