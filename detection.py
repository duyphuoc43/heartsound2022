import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
import pickle
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.compat.v2 as tf
import numpy as np


def get_data(name_files,audio,time_limit,sr):
    n_fft = 512
    n_mels= 128
    hop_length = 256
    segment_length = time_limit * sr
    num_segments = (len(audio) // segment_length)
    x = []
    y = []
    for i in range(num_segments):
        start_sample = i * segment_length
        end_sample = (i + 1) * segment_length
        
        segment = audio[start_sample:end_sample]

        create_data(segment,n_fft,n_mels,hop_length)
        
        label = name_files.split('/')[2]
        
        x.append(create_data(segment,n_fft,n_mels,hop_length))

        y.append(label)
    # print(len(x))
    return x,y
def create_data (audio,n_fft,n_mels,hop_length):
    output_size = (75,75,1)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,n_fft= n_fft ,n_mels=n_mels,hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram ,ref=np.max)
    mel_spectrogram_db = mel_spectrogram_db + np.abs(np.min(mel_spectrogram_db))
    mel_spectrogram_db= cv2.resize(mel_spectrogram_db,output_size[:2])
    mel_spectrogram_db= np.reshape(mel_spectrogram_db,output_size)
    return mel_spectrogram_db
def main(name_files):
    duration = 2
    time_limit = duration
    sr = 4000
    x = []
    x_test = []
    
    audio, sr = librosa.load(name_files, sr=sr,offset = 0)
    x,y = get_data(name_files, audio, time_limit = time_limit, sr = sr)
    x_test.extend(x)
    x_test = np.concatenate([x_test, x_test, x_test], axis=-1)
    model = load_model('model/final_model_DenseNet1691.h5')
    predictions = model.predict(x_test)
    for result in predictions:
        print(np.argmax(result))
if __name__ == "__main__":
    main("the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/2530_AV.wav")