
# coding: utf-8

# # d0td1 kernel (Laplace operator)
# ## imports
# 
# 

# In[1]:


#%matplotlib notebook

from numba import cuda, jit, double, void, prange, int32
from math import sqrt
from time import time
import numpy as np 

import matplotlib.pyplot as plt

import os

from ctypes import *


# In[2]:


# turn on numba warnings
os.environ["NUMBA_DEBUG_ARRAY_OPT_STATS"] = '1'
import numba
numba.__version__

# import cuda kernel
cwd = os.getcwd()
kernel = cdll.LoadLibrary(cwd+"/liblaplace3d_strides.so")

# load kernel from .cu

#laplace_3d_cpp = cuda.declare_device('laplace3d_strides','void(double[:,:,:], double[:,:,:],int32,int32,int32, int32,int32,int32)')
#set_val = cuda.declare_device('set_val','void(double[:,:,:])')

# get path to precompiled  library
curdir = os.path.join(os.path.dirname(__file__))
link = os.path.join(curdir, 'kernel.o')
print("linking: %s", link)

# ## Definitions

# In[3]:


# WNS factors 
eta_0 = 2./5.
eta_1 = 7./15.
eta_2 = 2./15.
eta_3 = 6./15.

alpha_0 = 5./6.
alpha_1 = 32./45.
alpha_2 = 2./45.
alpha_3 = 11./45.

cfln = 1.2/np.sqrt(3)

a = 1./24.*(1.-cfln)


# In[4]:


#@jit(void(double[:,:,:], double[:,:,:]), nogil=True, nopython=True, parallel=True)
#def laplace_3d_numpy(d, n):    
#    d[1:-1, 1:-1, 1:-1] = 1./2. * (n[:-2, 1:-1, 1:-1] + n[2:, 1:-1, 1:-1] +
#                          n[1:-1, :-2, 1:-1] + n[1:-1, 2:, 1:-1] +
#                          n[1:-1, 1:-1, :-2] + n[1:-1, 1:-1, 2:] - 6.*n[1:-1, 1:-1, 1:-1])


# In[5]:


@jit(void(double[:,:,:], double[:,:,:]), nogil=True, nopython=True, parallel=True)
def laplace_3d_loop(d, n):
    L = d.shape[0]
    M = d.shape[1]
    N = d.shape[2]
    
    for k in prange(1,N-1):
        for j in range(1,M-1):
            for i in range(1, L-1):
                d[i, j, k] = 1./2. * (n[i-1, j, k] + n[i+1, j, k] +
                                      n[i, j-1, k] + n[i, j+1, k] +
                                      n[i, j, k-1] + n[i, j, k+1] - 6.*n[i, j, k])


# In[6]:


#@cuda.jit(void(double[:,:,:], double[:,:,:]))
#def laplace_3d_cuda_kernel(d, n):
#    L = d.shape[0]
#    M = d.shape[1]
#    N = d.shape[2]
#    
#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
#    k = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
#    
#    
#    if j >= 1 and j < M - 1 and i >= 1 and i < L - 1 and k >= 1 and k < N - 1:
#        d[i, j, k] = 1./2. * (n[i-1, j, k] + n[i+1, j, k] +
#                                      n[i, j-1, k] + n[i, j+1, k] +
#                                      n[i, j, k-1] + n[i, j, k+1] - 6.*n[i, j, k])


#@cuda.jit()
def laplace_3d_cuda_opt_kernel(d,n, nx,ny,nz):  
    dd = cuda.to_device(d)
    dn = cuda.to_device(n)
    n=1
    
    cxt = cuda.current_context()
    cxt.synchronize()

    #for i in range(20):
    kernel.call_laplace3d_strides(dd.device_ctypes_pointer, dn.device_ctypes_pointer, nx,ny,nz) #.ctypes.data

    ts = time()
    for i in range(n):
        kernel.call_laplace3d_strides(dd.device_ctypes_pointer, dn.device_ctypes_pointer, nx,ny,nz) #.ctypes.data
    dt = time()-ts
    print(dt/n * 1000)
      
    dd.to_host()
    
    #print(d.ctypes.data)
    
def laplace_3d_cuda(d, n):
    L = d.shape[0]
    M = d.shape[1]
    N = d.shape[2]
    
    blockdim = (8,8,8)
    griddim = (L//blockdim[0], M//blockdim[1], N//blockdim[2])
    #print(griddim)
    
    stream = cuda.stream()
    dd = cuda.to_device(d,stream)
    dn = cuda.to_device(n,stream)
        
    #%timeit -n 32 -r 16 d0td1_cuda_kernel[griddim, blockdim](dd, dn)
    for i in range(0,100):
        laplace_3d_cuda_opt_kernel[griddim, blockdim,stream](dd, dn, L,M,N)
    
    evtstart = cuda.event(timing=True)
    evtend = cuda.event(timing=True)
    evtstart.record(stream)
    for i in range(100):
        laplace_3d_cuda_opt_kernel[griddim, blockdim,stream](dd, dn, L,M,N)
    evtend.record(stream);
    evtend.synchronize();
    print(cuda.event_elapsed_time(evtstart, evtend)/100.)
    
    dd.to_host()


@jit(nogil=True, nopython=True, parallel=True)
def prepare_input(nx,ny,nz, func, res_numpy, res_loop, res_cuda):
    func[:,:,:] = 0
    for i in prange(0,nx-1):
        for j in range(0,ny-1):
            for k in range(0,nz-1):
                func[i,j,k] = np.sin(i/(nx-1) * np.pi)*np.sin(j/(ny-1) * np.pi)*np.sin(k/(nz-1) * np.pi)
    
    res_numpy[:,:,:] = 0
    res_loop[:,:,:] = 0
    res_cuda[:,:,:] = 0


# In[8]:


def execute_kernels(res_numpy, res_loop, res_cuda, func, nx,ny,nz):
    n = 32*16
    #print("##### NUMPY #####")
    #ts = time()
    #for i in range(n):
    #    laplace_3d_numpy(res_numpy, func)    
    #dt = time()-ts
    #print(dt/n * 1000)
    
    print("##### LOOP #####")
    ts = time()
    for i in range(n):
        laplace_3d_loop(res_loop, func)    
    dt = time()-ts
    print(dt/n * 1000)
    
    print("##### GPU #####")
    # %timeit -n 32 -r 16 
    #laplace_3d_cuda(res_cuda, func)
    laplace_3d_cuda_opt_kernel(res_cuda, func, nx,ny,nz)

# In[9]:


def test_results(res_numpy, res_loop, res_cuda, func):
    #L(func) = -3*func
    print("##### TEST CASE #####")
    #residuum_numpy = - res_numpy/np.max(np.abs(res_numpy)) - func;
    #print('residuum_numpy =  ', repr(np.max(np.abs(residuum_numpy))))
   
    residuum_loop = - res_loop/np.max(np.abs(res_loop)) - func;
    print('residuum_loop = ', repr(np.max(np.abs(residuum_loop))))
    
    residuum_cuda = - res_cuda/np.max(np.abs(res_cuda)) - func;
    print('residuum_cuda = ', repr(np.max(np.abs(residuum_cuda))))

    #print("max difference numpy loop    = ", np.max(np.abs(res_numpy - res_loop)))
    #print("max difference numpy cuda    = ", np.max(np.abs(res_numpy - res_cuda)))

    #assert(np.max(np.abs(res_numpy - res_loop)) <= 10**-15)
    #assert(np.max(np.abs(res_numpy - res_cuda)) <= 10**-15)


# ## Execution

# In[10]:


nx = 512
ny = 512
nz = 128

func = np.zeros((nx,ny,nz), order='F')
res_numpy = np.zeros((nx,ny,nz), order='F')
res_loop = np.zeros((nx,ny,nz), order='F')
res_cuda = np.zeros((nx,ny,nz), order='F')


# In[11]:


prepare_input(nx, ny, nz, func, res_numpy, res_loop, res_cuda)


# In[12]:


#os.system("taskset -p 0xff %d" % os.getpid())
#laplace_3d_cuda_opt_kernel(res_cuda, func, nx,ny,nz)
#set_val_wrapper(func)

execute_kernels(res_numpy, res_loop, res_cuda, func,nx,ny,nz)
test_results(res_numpy, res_loop, res_cuda, func)

#plt.plot(func[:,50,50])
#plt.plot(-res_jit[:,50,50]/np.max(np.abs(res_jit)))
#plt.show()

