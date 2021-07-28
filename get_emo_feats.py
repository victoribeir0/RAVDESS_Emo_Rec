import numpy as np
import matplotlib.pylab as plt # Plot
import os
from scipy.io import wavfile
import time
import feature_ext as feat

def get_emo_feats(emo,TJan=40,Tav=10,ext='f0'):
    
    base = []    
    path = 'C:\\Users\\victo\\Documents\\Dataset _ EmoDB2\wav\\treino'
    files = os.listdir(path)

    for f in files:
        if f[5] == emo:

            ini = time.time()
            filename = path + '\\' + f
            Fs, x = wavfile.read(filename)      
            print('Inicialização: ' + str(time.time()-ini)) 

            ini = time.time()
            mat_esp = feat.get_spec(x, Fs, TJan, Tav, 0)    
            print('Espectro: ' + str(time.time()-ini))        

            ini = time.time()
            MFCC = feat.get_mfcc(mat_esp, Fs, TJan, TJan_dmfcc = 5)   
            
            lim = np.mean(MFCC[0,:])
            inds = np.nonzero(MFCC[0,:] > lim)[0].tolist()
            MFCC = MFCC[1:-1,inds]
            print('MFCC: ' + str(time.time()-ini)) 

            # ini = time.time()
            F0 = feat.get_f0(x, Fs, TJan, Tav, np.array(inds))   
            # print('F0: ' + str(time.time()-ini))           

            base = np.append(base,np.mean(F0))
            
    return base.tolist()

f0 = get_emo_feats('W',TJan=40,Tav=10,ext='f0')