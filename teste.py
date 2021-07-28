import time
from numba import jit, njit

def teste(num):
    res = []
    for n in range(num):
        if n % 2 == 0:
            res.append(1)
        else:
            res.append(2)

    return res

jit_func = jit()(teste)
ini = time.time()
_ = jit_func(np.array(range(0,100000)))
print(time.time()-ini)