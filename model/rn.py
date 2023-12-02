import torch
import torch.nn as nn
from torch import Tensor

# code basic by https://github.com/buptLinfy/ZSE-SBIR

class RelationNetwork(nn.Module) :
    def __init__(self, anchor, dropout=0.1) :
        super(RelationNetwork, self).__init__()

        self.rn = nn.Sequential(
            nn.Linear(anchor * anchor, 1280, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1280, 160, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(160, 1, bias=True)
        )

    # return the distance as [0 ~ 1]
    def forward(self, x) :
        x = self.rn(x)
        x = torch.sigmoid(x)
        return x
    
    
def cos_similar(p: Tensor, q: Tensor) :
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    sim_matrix = torch.where(torch.isnan(sim_matrix), torch.full_like(sim_matrix, 0), sim_matrix)
    return sim_matrix