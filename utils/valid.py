import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.ap import calculate
from tqdm import tqdm

import time

# code basic by https://github.com/buptLinfy/ZSE-SBIR. 

def valid_cls(args, model, sample_valid_data, origin_valid_data):
    model.eval()
    torch.set_grad_enabled(False)

    print('loading image data')
    sample_dataload = DataLoader(sample_valid_data, batch_size=args.test_sample_batch, num_workers=args.num_workers, drop_last=False)
    print('loading sketch data')
    origin_dataload = DataLoader(origin_valid_data, batch_size=args.test_origin_batch, num_workers=args.num_workers, drop_last=False)

    dist_im = None
    all_dist = None
    for i, (sample, sample_label) in enumerate(tqdm(sample_dataload)):
        if i == 0:
            all_sample_label = np.asarray(sample_label)
        else:
            all_sample_label = np.concatenate((all_sample_label, np.asarray(sample_label)), axis=0)

        sample_len = sample.size(0)
        sample = sample.cuda()
        sample, sample_idxs = model(sample, None, 'test', only_sa=True)

        for j, (origin, origin_label) in enumerate(tqdm(origin_dataload)):
            
            if i == 0 and j == 0:
                all_origin_label = np.asarray(origin_label)
            elif i == 0 and j > 0:
                all_origin_label = np.concatenate((all_origin_label, np.asarray(origin_label)), axis=0)

            origin_len = origin.size(0)
            origin = origin.cuda()
            origin, origin_idxs = model(origin, None, 'test', only_sa=True)

            sample_temp = sample.unsqueeze(1).repeat(1, origin_len, 1, 1).flatten(0, 1).cuda()
            origin_temp = origin.unsqueeze(0).repeat(sample_len, 1, 1, 1).flatten(0, 1).cuda()

            feature_1, feature_2 = model(sample_temp, origin_temp, 'test')

            if j == 0:
                dist_im = - feature_2.view(sample_len, origin_len).cpu().data.numpy()  # 1*args.batch
            else:
                dist_im = np.concatenate((dist_im, - feature_2.view(sample_len, origin_len).cpu().data.numpy()), axis=1)

        if i == 0:
            all_dist = dist_im
        else:
            all_dist = np.concatenate((all_dist, dist_im), axis=0)

    class_same = (np.expand_dims(all_sample_label, axis=1) == np.expand_dims(all_origin_label, axis=0)) * 1
    map_all, map_200, precision100, precision200 = calculate(all_dist, class_same, test=True)

    return map_all, map_200, precision100, precision200
