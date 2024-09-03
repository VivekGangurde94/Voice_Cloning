# Voice_cloinig_assignment
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

