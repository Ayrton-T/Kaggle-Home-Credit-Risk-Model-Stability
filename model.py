import torch
import torch.nn as nn

class AttentionMLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0., pred=True):
        super().__init__()
        self.pred = pred
        self.q = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        
        self.fc2 = nn.Linear(hidden_features, 1) if pred else nn.Linear(hidden_features, in_features)
            
        self.drop = nn.Dropout() 
        
    def forward(self, x):
        x0 = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        
        attention = (q@k.transpose(-2, -1))
        attention = attention.softmax(dim=-1)
        x = (attention@v).squeeze(2)
        
        x += x0
        x1 = x
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.pred == False:
            x += x1
        
        x = x.squeeze(0)
        return x
       
        
class Transformer(nn.Module):
    def __init__(self, in_features, drop=0.):
        super.__init__()
        self.block1 = AttentionMLP(in_features=in_features, hidden_features=2*in_features, drop=drop, pred=False)
        self.block2 = AttentionMLP(in_features=in_features, hidden_features=2*in_features, drop=drop, pred=True)

    def forward(self, x):
        return self.block2(self.block1(x))