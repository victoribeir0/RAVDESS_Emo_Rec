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

def get_f0(Tjan, Tav, inds, Fs, filename):
    fs, x = wavfile.read(filename)
    N = x.shape[0]
    Njan = int((Tjan/1000)*Fs) # Num. de amostras em cada janela.
    NAv = int((Tav/1000)*Fs)   # Num. de amostras para o avanço (sobreposição).

    ini = round(Fs/500) # Isso permite que os f0 obtidos sejam < 500 Hz.
    fim = round(Fs/75)  # Isso permite que os f0 obtidos sejam > 75 Hz.
    R = np.zeros(shape = (inds.shape[0],len(range(1,fim+1)))) # Inicialização da matriz de autocorrelaçao.
    b = np.zeros(shape = (len(range(1,fim+1)), Njan))         # Inicialização da matriz de dados.

    for i in range(0,inds.shape[0]): # Laço for para cada janela específica.
        aux = inds[i] # Define a janela específica.
        ap = ((aux-1)*NAv)+1
        a = x[ap:ap+Njan]
        a = a-np.mean(a) # Remove o nível DC subtraindo pela média.

        # Constroi a matriz b, com os dados de x atrasos em m amostras.
        for m in range(ini,fim):
            if (ap+Njan-1)+m-1 <= N:
                med = np.mean(x[range(ap+m-1,(ap+Njan-1)+m)])
                b[m,:] = x[range(ap+m-1,(ap+Njan-1)+m)]-med

        R[i,:] = np.matmul(b,a) # % Vetor de autocorrelação, para cada janela i.

    max_ind = np.argmax(R,1) # Obtém as posições dos valores máximos.
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










