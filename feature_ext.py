"""
    Calcula a autocorrelação e o F0 (estimado) do sinal x.
    x = Sinal de entrada.
    Tjan = Tempo de cada janela em ms.
    Tav = Tempo de avanço em ms (sobreposição entre janelas).
    inds = Janelas a serem utilizados.
    Fs = Freq. de amostragem.
    
    F0 = vetor F0 estimado.
    Obs: O vetor de F0 é reduzido pela mediana, valores distantes são removidos.
"""

import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pylab as plt # Plot
from pylab import imshow
import scipy as sp
from math import log, exp, sqrt, cos, pi, ceil, floor
from numpy.linalg import inv
import numpy.matlib
import time

def get_f0(x, Fs, Tjan, Tav, inds):
    # Fs, x = wavfile.read(filename)
    N = x.shape[0]
    Njan = int((Tjan/1000)*Fs) # Num. de amostras em cada janela.
    NAv = int((Tav/1000)*Fs)   # Num. de amostras para o avanço (sobreposição).

    ini = round(Fs/500) # Isso permite que os f0 obtidos sejam < 500 Hz.
    fim = round(Fs/75)  # Isso permite que os f0 obtidos sejam > 75 Hz.
    R = np.zeros(shape = (inds.shape[0],len(range(1,fim+1)))) # Inicialização da matriz de autocorrelaçao.
    b = np.zeros(shape = (len(range(1,fim+1)), Njan))         # Inicialização da matriz de dados.

    init = time.time()
    for i in range(0,inds.shape[0]): # Laço for para cada janela específica.
        
        aux = inds[i] # Define a janela específica.
        ap = ((aux)*NAv)+1
        a = x[ap:ap+Njan]
        a = a-np.mean(a) # Remove o nível DC subtraindo pela média.

        # Constroi a matriz b, com os dados de x atrasos em m amostras.
        for m in range(ini,fim):
            if (ap+Njan-1)+m-1 <= N:
                med = np.mean(x[range(ap+m-1,(ap+Njan-1)+m)])
                b[m,:] = x[range(ap+m-1,(ap+Njan-1)+m)]-med

        
        R[i,:] = np.matmul(b,a) # % Vetor de autocorrelação, para cada janela i.
    print('Mat mult F0:' + str(time.time()-init))

    max_ind = np.argmax(R,1) # Obtém as posições dos valores máximos.
    inds_zero = np.nonzero(max_ind==0)[0].tolist()
    max_ind = np.delete(max_ind,inds_zero) # Remove os zeros, caso exista.

    F0 = Fs/max_ind          # Obtém o F0, Fs/max_pos.
    F0 = F0[F0 < 500]        # Mantém somente os menos que 500 Hz.
    F0 = F0[F0 > 75]         # Mantém somente os menos que 75 Hz.

    if not(F0.shape[0]%2):   # Caso seja par remove um termo, o menos provável de ser F0 (mais distnte da média).
        pos_rem = np.argmax(np.abs(F0-np.mean(F0)))
        F0 = np.delete(F0,pos_rem)

    mdn = np.median(F0)
    range_mdn = 15 # Valor em que a mediana pode variar (para mais ou para menos).

    F0 = F0[F0 >= mdn-range_mdn] # Atualiza os k0 (descarta os que estão distantes de mediana).
    F0 = F0[F0 <= mdn + range_mdn]

    return F0

"""
    Calcula o espectrograma de um sinal x.
    TJan = Tempo da janela (em ms).
    Tav = Tempo de avanço (em ms).
    plotimg = Plotar a espectro (1 = Sim, 0 = Não).    
"""

def get_spec(x, Fs, TJan, Tav, plotimg):

    N = x.shape[0]               # Núm. de amostras do sinal x.
    TJan = int((TJan / 1000)*Fs) # Comprimento de cada janela (am amostras).
    Tav = int((Tav/1000)*Fs)     # Num. de amostras para o avanço.
    NJan = int((N - TJan) / Tav) # Núm. total de janelas (incluindo as sobrepostas).

    mat_esp = np.zeros([int(TJan/2), NJan]) # Inicializa matriz de espectrograma.

    janHamming = 0.54-0.46*np.cos(2*np.pi*np.array(range(0,TJan))/TJan) # Janela de Hamming.
    x = x[0:-2]-0.97*x[1:-1]

    for n in range(NJan):
        apontador = int(n * Tav)
        y = x[apontador:apontador+TJan]
        y = y*janHamming    # Multiplica pela janela de Hamming.
        y = fft(y)          # Calcula FFT.
        y = np.abs(y)       # Calcula o abs. da FFT.
        y = y[0:len(y)//2]  # Obtém a metade da FFT.
        mat_esp[:, n] = y   # Guarda resultado na matriz de espectrograma.

    if plotimg == 1:
        imshow(mat_esp, interpolation = 'nearest')
        plt.show()

    return mat_esp

"""
    Determina a matriz MFCC e a delta MFCC de um sinal.
    mat_esp = Matriz de espectrograma.
    fs = Freq. de amostragem do sinal.
    tam_jan = Tam. da janela do MFCC, em ms (recomendado: 25 ms).
    jan_dmfcc = Tam. da janela do delta MFCC (sempre ímpar, recomendado: 3 ou 5).
"""

def get_mfcc(mat_esp, Fs, TJan, TJan_dmfcc):

    TJan = int((TJan / 1000)*Fs) # Comprimento de cada janela (am amostras).
    fmin = 133.333  # Freq. mín.
    fmax = Fs/2     # Freq. máx.
    nFiltLin = 13   # N de filtros lineares.
    deltaLin = ((1000-fmin)/(nFiltLin-1))  # Intervalos lineares.

    nFiltLog = 27 # N de filtros log.
    deltaLog = (log(fmax)-log(1000))/(nFiltLog-1) # Intervalos log.

    nFiltTotal = nFiltLin+nFiltLog # N° de filtros no total.

    escala = np.array(range(int(TJan/2)-1)) 
    escala = (escala*Fs)/TJan

    escalaMel = np.zeros([1,nFiltTotal+2])
    esc1 = np.array(range(nFiltLin))*deltaLin
    escalaMel[0][0:nFiltLin] = esc1+fmin

    razaoPg = exp(deltaLog)
    esc2 = np.array(range(nFiltLog+2))
    escalaMel[0][nFiltLin:nFiltTotal+2] = escalaMel[0][nFiltLin-1]*razaoPg**esc2
    
    freq_i = escalaMel[0][0:nFiltTotal]   # Freq. do início das bandas críticas
    # freq_c = escalaMel[0][1:nFiltTotal+1]
    freq_f = escalaMel[0][2:nFiltTotal+2] # Freq. do final das bandas críticas

    filtros = np.zeros([nFiltTotal,int(TJan/2)])

    # Deteminação dos filtros.
    for n in range(nFiltTotal):
        ind_i = (np.absolute(escala - freq_i[n])).argmin()
        ind_f = (np.absolute(escala - freq_f[n])).argmin()
        #filtros[n,ind_i:ind_f+1] = 1

        #Filtros retangulares.
        for col in range(ind_i,ind_f+1):
            filtros[n,col] = 1
        
        # Normalização dos filtros
        filtros[n,:] = filtros[n,:]*(1/(freq_f[n]-freq_i[n]))

    # Transformada do Coseno.
    numCoefCep = 19
    matDCT = np.zeros([numCoefCep,nFiltTotal])
    peso = 1/sqrt(numCoefCep)

    #aux = np.array(range(nFiltTotal))
    #b = np.array(range(0,numCoefCep))
    #bt = np.matlib.repmat(b,40,1)
    #bt = np.transpose(bt)
    #matDCT = peso*np.cos(pi*bt)*((aux-0.5)/nFiltTotal)
    
    for lin in range(numCoefCep):
        aux = np.array(range(nFiltTotal))
        matDCT[lin,:] = peso*cos(pi*lin)*((aux-0.5)/nFiltTotal)    

    # Obtém as energias logarítmicas.
    EspLog = np.log10(np.matmul(filtros,mat_esp)+10e-6)
    # Obtém a matriz MFCC.
    MFCC = np.matmul(matDCT,EspLog)
    ll,cc = MFCC.shape  # Obtém as dimensões da matriz MFCC.
    MFCC = MFCC[1:ll,:] # Remove a primeira linha da matriz de MFCC.
    dMFCC = dmfcc(MFCC,TJan_dmfcc)

    rem = floor(TJan_dmfcc / 2)
    MFCC = MFCC[:,rem:cc-rem]

    return np.vstack((MFCC,dMFCC))

# Calcula a matriz de delta cepstral.
def dmfcc(mat_mfcc,jan):
    lin,col = mat_mfcc.shape

    idxI = ceil(jan/2)-1
    idxF = col-idxI
    alcance = floor(jan/2)

    dm = np.zeros([lin,col-(idxI*2)])

    for i in range(lin):
        for j in range(idxI,idxF):
            vet = mat_mfcc[i,j-alcance:j+alcance+1]
            dm[i,j-idxI] = get_ang(vet)

    return dm

def get_ang(y):
    x = np.array(range(len(y)))
    vet1 = np.ones([1,len(y)])
    m = np.vstack((vet1[0],x))
    m = np.transpose(m)

    invm = inv(np.matmul(np.transpose(m),m))
    aux1 = np.matmul(invm,np.transpose(m))
    a = np.matmul(aux1,y)
    return a[1]