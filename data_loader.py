from options import Option
from torchvision.transforms import transforms

def preprocess(audio_path, audio_type='result') :
    transform = transforms.ToTensor()

    if audio_type == 'result' :
        n = 1