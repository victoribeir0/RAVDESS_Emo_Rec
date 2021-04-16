import numpy as np
import librosa # Bliblioteca para processamento e análise de áudio.
import matplotlib.pylab as plt # Plot
import os
from scipy.io import wavfile

def get_emo_feats(emo,TJan):
    base = []
    path = 'C:\\Users\\victo\Documents\Dataset _ EmoDB2\wav\\treino'
    files = os.listdir(path)
    TJan = round((TJan / 1000) * 22050)  # Define o tamanho de cada janela em amostras.

    for f in files:
        if f[5] == emo:
            filename = path + '\\' + f
            # fs, x = wavfile.read(filename)
            x, fs = librosa.load(filename)
            f0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=65, fmax=500, sr=fs, frame_length=TJan, win_length=None)
            f0 = f0[~np.isnan(f0)]

            base = np.append(base,np.mean(f0))

    return base.tolist()
