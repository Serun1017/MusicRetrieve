import argparse

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")

        # dataset 
        parser.add_argument('--data_path', type=str, default='./dataset')
        parser.add_argument('--audio_max_length', type=int, default=240) # second
        parser.add_argument('--audio_min_length', type=int, default=10) # second
        parser.add_argument('--audio_sampling_rate', type=int, default=16000) # default 16kHz

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
