from pydub import AudioSegment
import os
import random

def preprocess_origin_data(dir_path, out_path) :
    start_time = 0
    end_time = 240000

    for (root, _, files) in os.walk(dir_path) :
        for file in files :
            if '.wav' in file :
                file_path = os.path.join(root, file)
                audio = AudioSegment.from_wav(file_path)

                file_name = os.path.splitext(file)[0]
                output_folder = os.path.join(out_path, file_name)
                os.makedirs(output_folder, exist_ok=True)

                for i in range(10) :
                    segment = audio[start_time:end_time]
                    output_path = os.path.join(output_folder, f"{file}_{i}.wav")

                    segment.export(output_path, format="wav")

def preprocess_sample_data(dir_path, out_path) :
    for (root, _, files) in os.walk(dir_path) :
        for file in files :
            if '.wav' in file :
                file_path = os.path.join(root, file)
                audio = AudioSegment.from_wav(file_path)

                file_name = os.path.splitext(file)[0]
                output_folder = os.path.join(out_path, file_name)
                os.makedirs(output_folder, exist_ok=True)

                for i in range(10) :
                    start_time = random.randint(0, 230) * 1000
                    end_time = start_time + 10000
                    segment = audio[start_time:end_time]
                    output_path = os.path.join(output_folder, f"{file}_{start_time}_{end_time}.wav")

                    segment.export(output_path, format="wav")

if __name__ == "__main__" :
    dir_path = 'dataset/valid/origin'
    out_path = 'dataset/valid/origin'

    # preprocess_sample_data(dir_path, out_path)
    preprocess_origin_data(dir_path, out_path)