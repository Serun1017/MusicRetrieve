import torch
import torch.nn as nn

# code by https://github.com/buptLinfy/ZSE-SBIR

def triplet_loss(x, args):
    triplet = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
    sample_p = x[0:args.batch]
    origin_p = x[2 * args.batch:3 * args.batch]
    origin_n = x[3 * args.batch:]
    loss = triplet(sample_p, origin_p, origin_n)

    return loss


def rn_loss(predict, target):
    mse_loss = nn.MSELoss().cuda()
    loss = mse_loss(predict, target)

    return loss


def classify_loss(predict, target):
    class_loss = nn.CrossEntropyLoss().cuda()
    loss = class_loss(predict, target)

    return loss



