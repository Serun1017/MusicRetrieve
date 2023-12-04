# Music Retrieve System

The code based by ZSE-SBIR(https://github.com/buptLinfy/ZSE-SBIR)

# DataSet Structure

```
dataset
   | 
    - train
        |
         - origin
              |
               - label 1
                    | 
                      - label1_0.wav  # original 4 minutes audio
                      - label1_1.wav  # original 4 minutes audio
                            ~
               - label 2
                    |
                      - label2_0.wav  # original 4 minutes audio
                      - label2_1.wav  # original 4 minutes audio
                            ~
         - sample
             |
               - label 1
                    | 
                      - label1_0_10000.wav  # sample 10 seconds audio(randomly selected)
                      - label1_20000_30000.wav  # sample 10 seconds audio(randomly selected)
                            ~
               - label 2
                    |
                      - label2_10000_20000.wav  # sample 10 seconds audio(randomly selected)
                      - label2_13000_23000.wav  # sample 10 seconds audio(randomly selected)
                            ~
    - valid
        |
         - origin # same as train
         - sample # same as train
    - test
        |
         - origin # same as train
         - sample # same as train
```

# Model Structure

The model idea is based on ZSE-SBIR. It is the model that classify the sketch based on Image. [ZSE-SBIR](https://github.com/buptLinfy/ZSE-SBIR)

As Transformer Model is based on Attention, First, We must find the attention in original music and sample audio. 

Second, use cross-attention to compare both attention and find the relation between original music and sample audio.

In this method, I used waveform. (I think that is the reason that it failed or too slow to train.)

I tokenized the both data and put them into Encoder. The audio tokenize method is as below.

## Audio Tokenize Method

### Origin

|   | in_channel | out_channel | kernel_size | stride | padding |
| --- | --- | --- | --- | --- | --- |
| 1st Conv1d | 2 | 64 | 16,000 * 10 (10 seconds) | 16,000 * 0.625 (0.625 seconds) | 16,000 * 5 -1 |
| 2nd Conv1d | 64 | 256 | 16 (1 second) | 4 (0.25 seconds) | 8 |
| 3rd Conv1d | 256 | 768 | 4 (0.25 seconds) | 1 (0.25 seconds) | 2 |
| 4th Conv1d | 768 | 768 | 6 | 6 | 1 |

4th Conv1d for reduce token (960 tokens -> 160 tokens)

result: 160 tokens. (Each tokens embedded 0.25 seconds waveform.)

### Sample

|   | in_channel | out_channel | kernel_size | stride | padding |
| --- | --- | --- | --- | --- | --- |
| 1st Conv1d | 2 | 256 | 16,000 * 1 (1 second) | 16,000 * 0.625 (0.625 seconds) | 16,000 * 0.5 |
| 2nd Conv1d | 256 | 768 | 4 (0.25 seconds) | 1 (0.25 seconds) | 1 |

result : 160 tokens. (Each tokens embedded 0.25 seconds waveform.)

# Result (In Progress...)

| epoch | loss | tri_loss | rn |
| --- | --- | --- | --- |
| 1 | 2.830 | 1.869 | 0.961 |
| 2 | 2.936 | 1.975 | 0.961 | 
| 3 | 3.352 | 2.392 | 0.960 |
| 4 | 3.188 | 2.226 | 0.963 |
| 5 | 3.049 | 2.084 | 0.965 |
| 6 | 3.109 | 2.149 | 0.959 |
| 7 | 2.974 | 2.009 | 0.965 |
| 8 | 2.919 | 1.959 | 0.960 |
| 9 | 2.655 | 1.698 | 0.957 |
| 10 | 2.774 | 1.813 | 0.961 |

The Loss had increased until epoch 6, but after epoch 6, the loss had beeing decreased.

I think it shows that the model find the relation after epoch 6, and it will find the relation as epoch increase.

But lack of time and errors in valid function, I couldn't finish it. (Please someone check whether is it work. I can't use GCP anymore...)

# Issue

Valid function has error when valid the test data. The tensor size is not fitted.

You can run the valid function in `test_valid.py`

Please leave comment when you find how to fix the Issue or error in `README.md`.