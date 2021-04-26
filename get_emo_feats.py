import numpy as np
import matplotlib.pylab as plt # Plot
import os
from scipy.io import wavfile

import feature_ext as feat

def get_emo_feats(emo='W',TJan=40,Tav=10,ext='f0'):
    base = []    
    path = 'C:\\Users\\victo\\Documents\\Dataset _ EmoDB2\wav\\treino'
    files = os.listdir(path)

    for f in files:
        if f[5] == emo:
            filename = path + '\\' + f
            Fs, x = wavfile.read(filename)      

            mat_esp = feat.get_spec(x, Fs, TJan, Tav, 0)
            MFCC = feat.get_mfcc(mat_esp, Fs, TJan, TJan_dmfcc = 5)    
            lim = np.mean(MFCC[0,:])
            inds = np.nonzero(MFCC[0,:] > lim)[0].tolist()
            MFCC = MFCC[1:-1,inds]

            F0 = feat.get_f0(x, Fs, TJan, Tav, np.array(inds))            

            base = np.append(base,np.mean(F0))

    return base.tolist()