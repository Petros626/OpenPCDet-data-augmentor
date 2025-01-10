import numpy as np 
from timeit import default_timer as timer 
from numba import jit, cuda
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# normal function to run on cpu 
def func(a):								 
	for i in range(10000000): 
		a[i]+= 1	

# function optimized to run on gpu
#@jit(target_backend='cuda')
@cuda.jit						 
def func2(a):
	idx = cuda.grid(1)
	if idx < a.size:
		a[idx] +=1
		
if __name__=="__main__": 
    n = 10000000							
    a = np.ones(n, dtype = np.float64) 

    start = timer() 
    func(a) 
    print("without GPU:", timer()-start)

    d_a = cuda.to_device(a)  # Daten auf die GPU übertragen

    blockdim = 1024  # Blockgröße
    griddim = (n + (blockdim - 1)) // blockdim

    start = timer()
    func2[griddim,blockdim](d_a)
    d_a.copy_to_host(a)
    print("with GPU:", timer()-start)