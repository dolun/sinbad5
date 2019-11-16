import sys
import os
import numba as nb
from numba import njit
import numpy as np
import math
import pylab as pl
from pylab import (arange, transpose, clip,  # ,scatter,show#,shape,cos,pi,reshape,dot,zeros
                   argsort, array, c_,  empty, exp, float32, int32,
                   float64, hstack, int32, linspace, load, log, log10,
                   logical_and, logspace, meshgrid,  ones_like, floor, ceil,
                   poisson, poly1d, polyfit, r_, rand, randn, ravel, real,
                   sqrt, subplots, uniform, unique, zeros, zeros_like, loadtxt, where)

import threading

NBSIGMA = 6
M_SQRT1_2 =1/sqrt(2)

@njit(nb.void(nb.int64, nb.float64, nb.float64, nb.float64[:], nb.int64, nb.int64,
              nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
      nogil=True, parallel=False)
def allocbin(nbi, kevcan, offset, spectre, istart, istop,
             weights_pics, location_pics, std_pics, weights_background, retour):

    l_pics = len(weights_pics)
    nbcanaux = istop-istart
    proba_pics = empty((nbcanaux, l_pics), dtype=float64)
    id_pics = empty((nbcanaux, l_pics), dtype=int32)
    n_pics = empty(nbcanaux, dtype=int32)
    inv_kevcan = kevcan
    for ipic in range(l_pics):
        dk = location_pics[ipic]
        std_pic=std_pics[ipic]
        ds = M_SQRT1_2/std_pic
        i_g = int(floor((dk-NBSIGMA*std_pic-offset)*inv_kevcan))
        i_d = int(ceil((dk+NBSIGMA*std_pic-offset)*inv_kevcan))
        if i_g < istart:
            i_g = istart
        if i_d > istop:
            i_d = istop
        if i_g<=i_d: continue

        energ=offset+kevcan*i_g-dk
        cvd= math.erf(energ*ds)
        pds=.5*weights_pics[ipic]
        for i in arange(i_g,i_d):
            energ+=kevcan
            nextcvd=math.erf(energ*ds)
            i0=i-istart
            n=n_pics[i0]
            proba_pics[i0][n]=(nextcvd-cvd)*pds
            id_pics[i0][n]=ipic
            n_pics[i0]=n_pics[i0]+1




@njit(nb.void(nb.int64, nb.float64[:], nb.float64[:]), nogil=True, parallel=False)
def dumb(nsamples, tab, retour):
    acc = 0
    for _ in nb.prange(nsamples):
        x = pl.rand()
        y = pl.rand()
        if (x**2 + y**2) < 1.0:
            acc += 1
    # return 4.0 * acc / nsamples
    retour[0] = 4.0 * acc / nsamples


def allocbin_mt(numthreads):
    arg = arange(8192, dtype=float64)
    length = len(arg)
    chunklen = (length + numthreads - 1) // numthreads
    chunks = [arg[i * chunklen:(i + 1) * chunklen] for i in range(numthreads)]
    ret = array([0.]*numthreads)
    threads = [threading.Thread(target=dumb, args=(1e8, chunk, ret[idthread:idthread+1]))
               for idthread, chunk in enumerate(chunks)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # print(ret)
    return ret
