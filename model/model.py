import torch
import torch.nn as nn
from .sa import SelfAttention
from .ca import CrossAttention
from .rn import RelationNetwork

class Model(nn.Module) :
    def __init__(self, args):
        self.args = args
        
        self.self_attention_origin = SelfAttention(args, type='origin')
        self.self_attention_sample = SelfAttention(args, type='sample')

        self.cross_attention = CrossAttention(args)

        self.relation_network = RelationNetwork(args)


    def forward(self, sample_of_music, origin_music, stage='train') :

        # sa_feas of each music
        sample_of_music = self.self_attention_sample(sample_of_music)
        origin_music = self.self_attention_origin(origin_music)

        ca_fea = self.cross_attention(sample_of_music, origin_music)

        

        return