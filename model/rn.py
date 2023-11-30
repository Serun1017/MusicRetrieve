import torch
import torch.nn as nn

class RelationNetwork(nn.Module) :
    def __init__(self, args) :
        super(RelationNetwork, self).__init__()

        self.rn = nn.Sequential()

    # return the distance as [0 ~ 1]
    def forward(self, x) :
        x = self.rn(x)
        x = torch.sigmoid(x)
        return x
    
    