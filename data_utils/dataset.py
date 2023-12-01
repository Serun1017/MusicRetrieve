import os
import numpy as np
import torch
import torchaudio
import copy
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import DataLoader
from torchaudio import transforms

def load_data_test(args) :
    sample_valid_data = ValidSet(args.valid_sample_data, type_skim='sample', half=True)
    origin_valid_data = ValidSet(args.valid_origin_data, type_skim='origin', half=True)
    return sample_valid_data, origin_valid_data

def load_data(args) :
    train_data = TrainSet(sample_dir_path=args.train_sample_data, origin_dir_path=args.train_origin_data)
    sample_valid_data = ValidSet(args.valid_sample_data, type_skim='sample', half=True)
    origin_valid_data = ValidSet(args.valid_origin_data, type_skim='origin', half=True)
    return train_data, sample_valid_data, origin_valid_data

class TrainSet(data.Dataset) :
    def __init__(self, sample_dir_path, origin_dir_path) :
        self.sample_data = ValidSet(sample_dir_path, type_skim='sample')
        self.origin_data = ValidSet(origin_dir_path, type_skim='origin')
        self.sample_data_neg = copy.deepcopy(self.sample_data)
        self.origin_data_neg = copy.deepcopy(self.origin_data)

        self.origin_data_neg.roll()

    def __getitem__(self, index):
        sample_data, sample_label = self.sample_data.__getitem__(index)
        origin_data, origin_label = self.origin_data.__getitem__(index)
        sample_data_neg, sample_label_neg = self.sample_data_neg.__getitem__(index)
        origin_data_neg, origin_label_neg = self.origin_data_neg.__getitem__(index)

        return sample_data, origin_data, sample_data_neg, origin_data_neg, \
               sample_label, origin_label, sample_label_neg, origin_label_neg
    
    def __len__(self) :
        return len(self.sample_data)

class ValidSet(data.Dataset) :
    def __init__(self, dir_path, half=False, type_skim='origin') :     
        self.file_names = []
        self.cls = []
        self.half = half
        self.type_skim = type_skim

        for (root, _, files) in os.walk(dir_path) :
            for file in files :
                if '.wav' in file :
                    file_path = os.path.join(root, file)
                    cls = os.path.basename(os.path.dirname(file_path))

                    self.file_names = np.append(self.file_names, [file_path], axis=0)
                    self.cls = np.append(self.cls, [cls], axis=0)

    def __getitem__(self, index) :
        label = self.cls[index]
        file_name = self.file_names[index]

        if self.half :
            if self.type_skim == 'sample' :
                audio = preprocess(file_name, noise=True).half()
            else :
                audio = preprocess(file_name, noise=False).half()
        else :
            if self.type_skim == 'sample' :
                audio = preprocess(file_name, noise=True)
            else :
                audio = preprocess(file_name, noise=False)
        return audio, label
    
    def __len__(self) :
        return len(self.file_names)
    
    def roll(self) :
        self.file_names = np.roll(self.file_names, shift=-1)
        self.cls = np.roll(self.cls, shift=-1)

def preprocess(file_name, noise=False) :
    waveform, sampling_rate = torchaudio.load(file_name)
    if noise :
        noise = torch.rand_like(waveform)
        snr = 10
        transform = torch.nn.Sequential(
            transforms.Resample(sampling_rate, 16000)
            #### add noise
        )
    else :
        transform = torch.nn.Sequential(
            transforms.Resample(sampling_rate, 16000)
        )
    return transform(waveform)

if __name__ == '__main__' :
    validset = TrainSet('./test_wav', './test_wav')
    test_loader = DataLoader(validset, batch_size=10, num_workers=2, drop_last=False)

    for index, (sample, origin, sample_neg, origin_neg, sample_label, origin_label, sample_label_neg, origin_label_neg) in enumerate(tqdm(test_loader)) :
        print(sample.shape)
        print(origin.shape)
        print(sample_neg.shape)
        print(origin_neg.shape)
        print(sample_label)
        print(origin_label)
        print(sample_label_neg)
        print(origin_label_neg)
