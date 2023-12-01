import argparse

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")

        # dataset 
        parser.add_argument('--train_sample_data', type=str, default='./dataset')
        parser.add_argument('--train_origin_data', type=str, default='./dataset')
        parser.add_argument('--audio_sampling_rate', type=int, default=16000) # default 16kHz

        # valid set
        parser.add_argument('--valid_sample_data', type=str, default='./dataset')
        parser.add_argument('--valid_origin_data', type=str, default='./dataset')

        # test set
        parser.add_argument('--test_sample_data', type=str, default='./dataset')
        parser.add_argument('--test_origin_data', type=str, default='./dataset')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
