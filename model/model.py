import torch
import torch.nn as nn

from .sa import SelfAttention
from .ca import CrossAttention
from .rn import RelationNetwork

class Model(nn.Module) :
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        self.self_attention_origin = SelfAttention(args, type='origin')
        self.self_attention_sample = SelfAttention(args, type='sample')

        self.cross_attention = CrossAttention()

        self.relation_network = RelationNetwork(args)


    def forward(self, sample_of_music, origin_music, stage='train') :

        # sa_feas of each music [batch, token, channel]
        sa_fea_sample, left_tokens_sample, idxs_sampel = self.self_attention_sample(sample_of_music)
        sa_fea_origin, left_tokens_origin, idxs_origin = self.self_attention_origin(origin_music)

        ca_sample, ca_origin = self.cross_attention(sa_fea_sample, sa_fea_origin)
        
        return ca_sample, ca_origin