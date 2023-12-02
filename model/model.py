import torch
import torch.nn as nn

from .sa import SelfAttention
from .ca import CrossAttention
from .rn import RelationNetwork, cos_similar

# code basic by https://github.com/buptLinfy/ZSE-SBIR
class Model(nn.Module) :
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        self.self_attention_origin = SelfAttention(args, type='origin')
        self.self_attention_sample = SelfAttention(args, type='sample')

        self.cross_attention = CrossAttention()

        self.relation_network = RelationNetwork(160)


    def forward(self, sample_music, origin_music, stage='train', only_sa=False) :
        if stage == 'train' :
            # sa_feas of each music [batch, token, channel]
            sa_fea_sample, left_tokens_sample, idxs_sampel = self.self_attention_sample(sample_music)
            sa_fea_origin, left_tokens_origin, idxs_origin = self.self_attention_origin(origin_music)

            ca_fea = self.cross_attention(sa_fea_sample, sa_fea_origin)

            cls_fea = ca_fea[:, 0]
            token_fea = ca_fea[:, 1:]
            batch = token_fea.size(0)

            token_fea = token_fea.view(batch, 768, 160)
            token_fea = token_fea.transpose(1, 2) # [batch * 2(sample + origin), token, channel]
            
            sample_fea = token_fea[:batch // 2]
            origin_fea = token_fea[batch // 2:]

            cos_scores = cos_similar(sample_fea, origin_fea)
            cos_scores = cos_scores.view(batch // 2, -1)
            rn_scores = self.relation_network(cos_scores)

            return cls_fea, rn_scores
        else :
            if only_sa :
                if sample_music is not None :
                    sample_sa_fea, _, sample_idxs = self.self_attention_sample(sample_music)
                    return sample_sa_fea, sample_idxs
                
                if origin_music is not None :
                    origin_sa_fea, _, origin_idxs = self.self_attention_origin(origin_music)
                    return origin_sa_fea, origin_idxs
            
            else :
                ca_fea = self.cross_attention(sample_music, origin_music)

                cls_fea = ca_fea[:, 0]
                token_fea = ca_fea[:, 1:]
                batch = token_fea.size(0)

                token_fea = token_fea.view(batch, 768, 160)
                token_fea = token_fea.transpose(1, 2) # [batch * 2(sample + origin), token, channel]
                
                sample_fea = token_fea[:batch // 2]
                origin_fea = token_fea[batch // 2:]

                cos_scores = cos_similar(sample_fea, origin_fea)
                cos_scores = cos_scores.view(batch // 2, -1)
                rn_scores = self.relation_network(cos_scores)
            
                return cls_fea, rn_scores
