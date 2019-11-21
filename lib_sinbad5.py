import sys
import os
import numba as nb
from numba import njit, prange
import numpy as np
import math
import pylab as pl
from pylab import (arange, transpose, clip, randint, gamma,  # ,scatter,show#,shape,cos,pi,reshape,dot,zeros
                   argsort, array, c_,  empty, exp, float32, int32, multinomial,
                   float64, hstack, int32, linspace, load, log, log10, log2,
                   logical_and, logspace, meshgrid,  ones_like, floor, ceil,
                   poisson, poly1d, polyfit, r_, rand, randn, ravel, real, ones,
                   sqrt, subplots, uniform, unique, zeros, zeros_like, loadtxt, where)

import threading

NBSIGMA = 6
M_SQRT1_2 = 1/sqrt(2)


@njit(['float64[:](float64[:],int64)',
       'float64[:](int64[:],int64)'], parallel=True, cache=True)
def rehistogram_pow_2(tab_in, lg_out):
    """ rehistogrammization on a power-2 number of bins """
    lg_in = len(tab_in)
    # assert lg_out <= lg_in, 'rehistogram_pow_2 error len'
    tab_out = empty(lg_out, dtype=float64)
    fr = lg_in/lg_out  # float
    for i in prange(lg_out):
        v1 = fr * i
        a = int(floor(v1))
        v2 = v1 + fr
        b = int(floor(v2))
        if b == lg_in:
            b -= 1
        tab_out[i] = tab_in[a] * (a + 1 - v1) + tab_in[a + 1] + tab_in[b] * (v2 - b) \
            if a+2 == b else \
            tab_in[a] * (b - v1) + tab_in[b] * (v2 - b)
    return tab_out


@njit(nb.float64[:](nb.int64[:], nb.float64, nb.float64, nb.int64[:], nb.float64),
      parallel=True, cache=True)
def polya_parallel(data, _a_polya, _p_polya, prior_polya_in, prop_prior_in):
    """ Polya Draw
        mean prior_polya_in = 1
    """
    lgin = len(data)  # len of histo
    log2_lgin = int(floor(log2(lgin)))
    lg_rehist = 1 << log2_lgin
    exp_prior_in = ones(lgin)

    tab_rehist_in = rehistogram_pow_2(data, lg_rehist)
    prior_polya_rehist = rehistogram_pow_2(prior_polya_in, lg_rehist)
    exp_prior_rehist = rehistogram_pow_2(exp_prior_in, lg_rehist)

    shift = randint(0, lg_rehist+1)
    # DOUBLEMENT DU SUPPORT
    lg_double = lg_rehist << 1
    log2_lg_double = log2_lgin+1

    tab_in = zeros(lg_double, dtype=float64)
    pr = zeros(lg_double, dtype=float64)

    # polya weight bottom
    aexpnp = _a_polya*exp(_p_polya*log2_lg_double)
    for j in prange(lg_rehist):
        tab_in[j + shift] = tab_rehist_in[j]
        pr[j + shift] = aexpnp * (prop_prior_in * prior_polya_rehist[j] +
                                  (1 - prop_prior_in) * exp_prior_rehist[j])

    lp = lg_double * 2 - 2
    weights = empty(lp)
    j = 0
    expmp = .5*exp(-_p_polya)
    q = 1
    while log2_lg_double > 0:
        n = 1 << log2_lg_double
        fin = n
        for i in arange((shift >> q) << 1, fin, 2):
            i_plus_one = i + 1
            weights[j + i] = tab_in[i] + pr[i]
            weights[j + i_plus_one] = tab_in[i_plus_one] + pr[i_plus_one]
            tab_in[i >> 1] = tab_in[i] + tab_in[i_plus_one]
            pr[i >> 1] = (pr[i] + pr[i_plus_one]) * expmp
        j += n
        q += 1
        log2_lg_double -= 1

    frac = zeros(lp)
    for half_k in prange(lg_double-1):
        k = half_k << 1
        wg, wd = weights[k], weights[k+1]
        if wg+wd > 0.:
            gauche = gamma(wg) if wg > 0. else 0.
            droite = gamma(wd) if wd > 0. else 0.
            frac[k] = gauche / (droite + gauche)
            frac[k + 1] = 1. - frac[k]

    loglg = 2
    j = lp - 2
    while j > lg_double:
        n = 1 << loglg
        j -= n
        for half_k in prange(n >> 1):
            k = half_k << 1
            v = frac[j + n + (k >> 1)]
            frac[j + k] *= v
            frac[j + k + 1] *= v

        loglg += 1

    tab_out_rehist = empty(lg_rehist, dtype=float64)
    for i in prange(lg_rehist):
        i_shift = i + shift
        tab_out_rehist[i] = frac[i_shift] * frac[lg_double + (i_shift >> 1)]

    # back to input binning
    tab_out = empty(lgin, dtype=float64)
    fr = lg_rehist/lgin  # float64
    for i in prange(lgin):
        v1 = fr * i
        v2 = v1 + fr
        a = int(floor(v1))
        b = int(floor(v2))
        if b == lg_rehist:
            b -= 1

        tab_out[i] = tab_out_rehist[a] * (a + 1 - v1) + \
            tab_out_rehist[a + 1] + \
            tab_out_rehist[b] * (v2 - b) if a + 2 == b else \
            tab_out_rehist[a] * (b - v1) + tab_out_rehist[b] * (v2 - b)

    return tab_out

@njit(nb.float64[:](nb.int64, nb.int64[:], nb.float64, nb.float64, nb.int64[:], nb.float64),
      parallel=False, cache=True)
def iterations(n_per_call, data, _a_polya, _p_polya, prior_polya_in, prop_prior_in):
    for _ in range(n_per_call):
        ret = polya_parallel(data, _a_polya, _p_polya,
                             prior_polya_in, prop_prior_in)
    return ret


@njit(nb.void(nb.float64, nb.float64, nb.int64[:], nb.int64, nb.int64,
              nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
      nogil=True, parallel=False)
def allocbin(kevcan, offset, spectre, istart, istop,
             weights_pics, location_pics, std_pics, weights_background, stats):
    # stats must be zeros

    l_pics = len(weights_pics)
    nbcanaux = istop-istart
    proba_pics = empty((nbcanaux, l_pics+1), dtype=float64)
    id_pics = empty((nbcanaux, l_pics+1), dtype=int32)
    n_pics = zeros(nbcanaux, dtype=int32)
    inv_kevcan = kevcan
    for ipic in range(l_pics):
        dk = location_pics[ipic]
        std_pic = std_pics[ipic]
        ds = M_SQRT1_2/std_pic
        i_g = int(floor((dk-NBSIGMA*std_pic-offset)*inv_kevcan))
        i_d = int(ceil((dk+NBSIGMA*std_pic-offset)*inv_kevcan))
        if i_g < istart:
            i_g = istart
        if i_d > istop:
            i_d = istop
        if i_g >= i_d:
            continue

        energ = offset+kevcan*i_g-dk
        cvd = math.erf(energ*ds)
        pds = .5*weights_pics[ipic]
        for i in arange(i_g, i_d):
            energ += kevcan
            nextcvd = math.erf(energ*ds)
            i0 = i-istart
            n = n_pics[i0]
            proba_pics[i0][n] = (nextcvd-cvd)*pds
            id_pics[i0][n] = ipic
            n_pics[i0] += 1

    for i in arange(istart, istop):
        i0 = i-istart
        n = n_pics[i0]
        if n > 0:
            proba_pics[i0][n] = weights_background[i]  # background
            prob = proba_pics[i0][:n+1]
            prob /= np.sum(prob)
            ret = multinomial(spectre[i], prob)
            bi0 = offset+kevcan*i
            bi1 = bi0+kevcan
            enmean = .5*(bi0+bi1)
            envar = (bi0*bi0+bi0*bi1+bi1*bi1)/3.
            for k in range(n):  # pics
                if ret[k] > 0:
                    indpic = id_pics[i0][k]
                    n_float = float(ret[k])
                    # sufficient statistics
                    stats[indpic] += n_float  # X**0
                    stats[indpic+l_pics] += n_float*enmean  # X**1
                    stats[indpic+2*l_pics] += n_float*envar  # X**2
            stats[i+3*l_pics] = ret[n]  # background

        else:  # all for background if no pic candidat
            stats[3*l_pics+i] = float(spectre[i])


def allocbin_mt(kevcan, offset, spectre,
                weights_pics, location_pics, std_pics, weights_background, nbthreads):
    l_pics = len(weights_pics)
    l_spectre = len(spectre)
    statth = zeros((nbthreads, 3*l_pics+l_spectre), dtype=float)

    tab_chunk = array((arange(nbthreads+1)*l_spectre)/nbthreads, dtype=int)
    threads = [threading.Thread(target=allocbin,
                                args=(kevcan, offset, spectre,
                                      tab_chunk[idthread], tab_chunk[idthread+1],
                                      weights_pics, location_pics, std_pics,
                                      weights_background, statth[idthread]))
               for idthread in range(nbthreads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return statth.sum(0)


"""
@njit(nb.void(nb.int64, nb.int64, nb.float64[:], nb.float64[:]),
      nogil=True, parallel=False)
def dumb(idthread, nsamples, tab, retour):
    acc = 0
    for _ in nb.prange(nsamples):
        x = np.random.rand()
        y = np.random.rand()
        if (x**2 + y**2) < 1.0:
            acc += 1
    # return 4.0 * acc / nsamples
    retour[0] = 4.0 * acc / nsamples


def dumb_mt(numthreads):
    arg = arange(8192, dtype=float64)
    length = len(arg)
    chunklen = (length + numthreads - 1) // numthreads
    chunks = [arg[i * chunklen:(i + 1) * chunklen] for i in range(numthreads)]
    ret = array([0.]*numthreads)
    threads = [threading.Thread(target=dumb, args=(idthread, 1e8, chunk, ret[idthread:idthread+1]))
               for idthread, chunk in enumerate(chunks)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return ret
"""

# ret = empty(1)
# a = arange(4, dtype=float)
# dumb(0, 10, a, ret)
