# Voice_cloinig
 Text-Based Voice Cloning

This repository contains the Fine tunning code and tts code

# Dataset Preparation
The original 20-minute audio file was processed as follows:

Clipping: The audio was divided into smaller clips, each ranging from 3 to 11 seconds in length.
Resampling: All clips were converted to a sample rate of 22,050 Hz.
Noise Reduction: Background noise was removed from the clips using Audacity.
After processing, the clips were organized into two parts:

Training Data: Contains 103 clips.
Evaluation Data: Contains 18 clips.
For each clip, the following metadata was included:

Path: The file path to the audio clip.
Transcription: The corresponding text transcription of the audio.
Speaker Name: The name of the speaker.
This dataset is now ready for use in training and evaluation. 
[**Dataset**](https://drive.google.com/drive/folders/1P_RGV_PgIu3esyAUdIudGK1UGe_NRvok?usp=sharing)

# Model Training
The model was fine-tuned using the following setup:

1. Training Configuration
Language: en
Number of Epochs: 30
Batch Size: 2
Gradient Accumulation Steps: 252
Maximum Audio Length: 255,995 samples (~11.6 seconds)
Optimizer: AdamW
Betas: [0.9, 0.96]
Epsilon: 1e-8
Weight Decay: 1e-3
Learning Rate: 1e-5
Scheduler: MultiStepLR with milestones at [50000 * 18, 150000 * 18, 300000 * 18] and gamma = 0.5
2. Dataset Configuration
Training Data: 103 audio clips
Evaluation Data: 18 audio clips
Sample Rate: 22,050 Hz for input, 22,050 Hz for output
Max Conditioning Length: 132,300 samples (6 seconds)
Min Conditioning Length: 66,150 samples (3 seconds)
Max Text Length: 200 characters
3. Model Files
The following model files were downloaded and used during training:

DVAE Checkpoint: dvae.pth
Mel-Norm File: mel_stats.pth
XTTS Checkpoint: model.pth
Tokenizer File: vocab.json

# Training Process
The model was trained using the fine-tuned parameters, with logs and checkpoints managed using TensorBoard. The training process involved loading the dataset, initializing the model with the GPTTrainerConfig, and using a customized Trainer class to fit the model.

