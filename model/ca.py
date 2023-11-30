import torch.nn as nn

class CrossAttention(nn.Module) :
    def __init__(self, argss) :
        super(CrossAttention, self).__init__()

    def forward(self, sample_sa_fea, origin_sa_fea) :
        return # something...