import torch
import torch.nn as nn
from .sa import Scale_Embedding

class Model(nn.Module) :
    def __init__(self, args):
        self.args = args
        

    def forward(self, sample_of_music, retrieve_music, stage='train') :
        return