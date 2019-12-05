import sys
import os
import numba as nb
from numba import njit, prange, threading_layer
from numba import cuda
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
from time import time, time_ns

NBSIGMA = 4
M_SQRT1_2 = 1/sqrt(2)
MAX_PICS = 1000


@njit(nb.int32(nb.int32, nb.float64))
def binomial_knuth(n, p):
    # return int(np.round(n*p))
    k = 0
    while n > 10:  # /* This parameter is tunable */
        a = 1 + n//2
        b = 1 + n - a
        xa = pl.gamma(a)
        xb = pl.gamma(b)
        X = xa/(xa+xb)
        # X = pl.beta(a, b)
        if X >= p:
            n = a - 1
            p /= X
        else:
            k += a
            n = b - 1
            p = (p - X) / (1 - X)

    for _ in range(n):
        if rand() < p:
            k += 1

    return k


@njit(nb.int32[:](nb.int32, nb.float64[:]))
def multinomial_knuth(N, p):
    sum_p = 0.
    sum_n = 0
    K = len(p)
    ret = empty(K, dtype=np.int32)
    norm = p.sum()

    for k in range(K):
        ret[k] = binomial_knuth(N-sum_n, p[k]/(norm-sum_p)) if p[k] > 0 else 0
        sum_p += p[k]
        sum_n += ret[k]
    return ret


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


@njit(nb.float64[:](nb.int64[:], nb.float64, nb.float64, nb.float64[:], nb.float64),
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


@njit
def get_var_from_energ(_e, _offset_variance_energy, _coef_variance_energy):
    return _offset_variance_energy+_e*_coef_variance_energy


@njit(nb.int64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64, nb.float64, nb.float64, nb.float64, nb.float64,
                  nb.float64[:], nb.float64[:], nb.float64[:]))
def param_update(tab_s1, tab_sx, tab_sx2, tab_sigma, offset, c1, alpha_dirichlet, emin, emax,
                 pics_weights, pics_locations, pics_variances):
    """
    # sufficient statistics s1,sx,sx2

    # compute pics weights,locations and variances
    # return number of pics
    """
    id_pic = 0
    for s1, sx, sx2, sigma in zip(tab_s1, tab_sx, tab_sx2, tab_sigma):
        if s1 > 0:
            w1 = gamma(s1)
            pics_weights[id_pic] = w1  # weight
            location = sx/s1+randn()*sigma/sqrt(s1)
            pics_locations[id_pic] = location  # location
            pics_variances[id_pic] = get_var_from_energ(
                location, offset, c1)  # variance
            id_pic += 1
    nb_pics_aff = id_pic
    # weights normalisation + residu
    w1 = gamma(alpha_dirichlet)
    inv_sum_weights = 1./(pics_weights[:id_pic].sum()+w1)
    pics_weights[:id_pic] *= inv_sum_weights
    residu = w1*inv_sum_weights

    len_residu = int(10.*alpha_dirichlet)
    for i in range(len_residu):
        v = pl.beta(1, alpha_dirichlet) if i < len_residu-1 else 1.
        pics_weights[id_pic] = v*residu
        residu *= 1.-v
        location = pl.uniform(emin, emax)
        pics_locations[id_pic] = location  # location
        pics_variances[id_pic] = get_var_from_energ(
            location, offset, c1)  # variance
        id_pic += 1

    return array((id_pic, nb_pics_aff))

"""
@njit(nb.float64[:](nb.float64, nb.float64, nb.int64[:],
                    nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
      nogil=True, parallel=True)
def draw_labels(kevcan, offset, spectre,
                weights_pics, location_pics, std_pics, weights_background):
    # stats must be zeros
    l_pics = len(weights_pics)
    nbcanaux = len(spectre)
    proba_pics = empty((nbcanaux, l_pics+1), dtype=float64)
    id_pics = empty((nbcanaux, l_pics), dtype=np.int64)
    n_pics = zeros(nbcanaux, dtype=np.int64)
    inv_kevcan = 1./kevcan
    for ipic in range(l_pics):
        dk = location_pics[ipic]
        std_pic = std_pics[ipic]
        i_g = int(floor((dk-NBSIGMA*std_pic-offset)*inv_kevcan))
        i_d = int(ceil((dk+NBSIGMA*std_pic-offset)*inv_kevcan))
        if i_g < 0:
            i_g = 0
        if i_d > nbcanaux:
            i_d = nbcanaux
        if i_g >= i_d:
            continue

        ds = M_SQRT1_2/std_pic
        energ = offset+kevcan*i_g-dk
        cvd = math.erf(energ*ds)
        pds = .5*weights_pics[ipic]
        for i in arange(i_g, i_d):
            energ += kevcan
            nextcvd = math.erf(energ*ds)
            n = n_pics[i]
            proba_pics[i][n] = (nextcvd-cvd)*pds
            id_pics[i][n] = ipic
            n_pics[i] += 1
            cvd = nextcvd

    stats = zeros(3*l_pics+nbcanaux, dtype=np.float64)

    for i in range(nbcanaux):
        n = n_pics[i]
        if n > 0:
            proba_pics[i][n] = weights_background[i]  # background
            prob = proba_pics[i][:n+1]
            prob /= np.sum(prob)
            ret = multinomial(spectre[i], prob)
            bi0 = offset+kevcan*i
            bi1 = bi0+kevcan
            enmean = .5*(bi0+bi1)
            # envar = (bi0*bi0+bi0*bi1+bi1*bi1)/3.
            for k in range(n):  # pics
                if ret[k] > 0:
                    indpic = id_pics[i][k]
                    n_float = float(ret[k])
                    # sufficient statistics
                    stats[indpic] += n_float  # X**0
                    stats[indpic+l_pics] += n_float*enmean  # X**1
                    # stats[indpic+2*l_pics] += n_float*envar  # X**2
            stats[3*l_pics+i] = float(ret[n])  # background

        else:  # all for background if no pic candidat
            stats[3*l_pics+i] = float(spectre[i])

    return stats
"""

@njit(nb.void(nb.float64, nb.float64, nb.int64[:], nb.int64, nb.int64,
              nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
              nb.float64[:], nb.float64[:]),
      nogil=True, parallel=False)
def allocbin_thread(kevcan, offset, spectre, istart, istop,
                    weights_pics, location_pics, std_pics, weights_background,
                    reconstruction, stats):

    # l_pics0 = len(weights_pics)
    # _weights_pics = ones(250)
    # _location_pics = linspace(offset+10., offset+kevcan*len(spectre)-10., 250)
    # _std_pics = ones(250)
    # stats must be zeros
    l_pics = len(weights_pics)
    nbcanaux = istop-istart
    proba_pics = empty((nbcanaux, l_pics+1), dtype=float64)
    id_pics = empty((nbcanaux, l_pics+1), dtype=int32)
    n_pics = zeros(nbcanaux, dtype=int32)
    inv_kevcan = 1./kevcan

    for ipic in range(l_pics):
        dk = location_pics[ipic]
        std_pic = std_pics[ipic]
        i_g = int(floor((dk-NBSIGMA*std_pic-offset)*inv_kevcan))
        i_d = int(ceil((dk+NBSIGMA*std_pic-offset)*inv_kevcan))
        if i_g < istart:
            i_g = istart
        if i_d > istop:
            i_d = istop
        if i_g >= i_d:
            continue

        ds = M_SQRT1_2/std_pic
        energ = offset+kevcan*i_g-dk
        cvd = math.erf(energ*ds)
        pds = .5*weights_pics[ipic]
        for i in arange(i_g, i_d):
            energ += kevcan
            nextcvd = math.erf(energ*ds)
            i0 = i-istart
            n = n_pics[i0]
            p=(nextcvd-cvd)*pds
            proba_pics[i0][n] = p
            reconstruction[i]+=p
            id_pics[i0][n] = ipic
            n_pics[i0] += 1
            cvd = nextcvd

    for i in arange(istart, istop):
        i0 = i-istart
        n = n_pics[i0]
        if n > 0:
            proba_pics[i0][n] = weights_background[i]  # background
            prob = proba_pics[i0][:n+1]
            prob /= reconstruction[i]#np.sum(prob)
            ret = multinomial_knuth(spectre[i], prob)
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
            stats[3*l_pics+i] = float(ret[n])  # background
        else:  # all for background if no pic candidat
            stats[3*l_pics+i] = float(spectre[i])
    return


def allocbin_mt(offset_energy_canal, kevcan, spectre,
                weights_pics, location_pics, std_pics, weights_background,
                 reconstruction, nbthreads):
    l_pics = len(weights_pics)
    l_spectre = len(spectre)
    statth = zeros((nbthreads, 3*l_pics+l_spectre), dtype=np.float64)

    tab_chunks = array((arange(nbthreads+1)*l_spectre) /
                       nbthreads, dtype=np.int64)
    threads = [threading.Thread(target=allocbin_thread,
                                args=(kevcan, offset_energy_canal, spectre,
                                      tab_chunks[idthread], tab_chunks[idthread+1],
                                      weights_pics, location_pics, std_pics,
                                      weights_background, reconstruction, statth[idthread]))
               for idthread in range(nbthreads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return statth.sum(0)

@cuda.jit('void(float64,float64,float64[:], float64[:], float64[:], float64[:,:])')
def compute_gpu_proba2d(offset, kevcan, weights, locations, sigmas, out):

    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    il,ip=block_id,thread_id
    loc = locations[ip]
    weight = weights[ip]
    ds = M_SQRT1_2/sigmas[ip]

    energ_g = offset+kevcan*il-loc
    energ_d = energ_g+kevcan

    out[il][ip] = .5*weight*(math.erf(energ_d*ds)-math.erf(energ_g*ds))

# @njit(nb.float64[:](nb.int64, nb.int64[:], nb.int64, nb.float64, nb.float64[:],
#                     nb.float64[:], nb.float64[:], nb.float64[:], nb.float64,
#                     nb.float64, nb.float64[:], nb.float64, nb.float64, nb.float64,
#                      nb.float64, nb.float64, nb.float64),
#       parallel=False, cache=True)
def iterations(
        n_per_call, data, nb_pics, proportionPicsFond,
        compton, pics_weights, pics_locations, pics_variances,meanReconstruction,
        _a_polya, _p_polya, prior_polya_in, prop_prior_in,
        offset_energy_canal, kevcan, offset_variance_energy, coef_variance_energy,
        alpha_dirichlet):

    nb_bins = len(data)
    NUM_THREADS = nb.config.NUMBA_DEFAULT_NUM_THREADS
    if NUM_THREADS > 8:
        NUM_THREADS = 8
    # NUM_THREADS=1
    t1 = t2 = t3 = 0.
    # for _ in range(n_per_call):
    t0 = time()
    nbiter = 0
    #GPU****************
    # gpu_out = np.zeros((nb_bins,MAX_PICS), dtype=np.float64)

    #GPU****************
    meanReconstruction[:]=0.
    while time()-t0 < .5:
        nbiter += 1
        # draw labels
        _weights = pics_weights[:nb_pics]
        _locations = pics_locations[:nb_pics]
        _sigmas = sqrt(pics_variances[:nb_pics])

        lt = time_ns()

        # compute_gpu_proba2d[nb_bins,nb_pics](offset_energy_canal, kevcan,
        # _weights, _locations, _sigmas,gpu_out)

        reconstruction=np.copy(compton)
        sufficient_stats = array(allocbin_mt(offset_energy_canal, kevcan, data,
                                             _weights, _locations, _sigmas, compton,
                                             reconstruction, NUM_THREADS), dtype=np.float64)

        # l_pics, l_spectre = len(_weights), len(data)
        # statth = zeros((1, 3*l_pics+l_spectre), dtype=np.float64)
        # allocbin_thread(kevcan, offset_energy_canal, data,
        #                 0, l_spectre,_weights, _locations, _sigmas,
        #                 compton, statth[0])
        # sufficient_stats=statth[0]

        now = time_ns()
        t1 += now-lt

        s1 = sufficient_stats[:nb_pics]  # float64
        sx = sufficient_stats[nb_pics:nb_pics*2]  # float64
        sx2 = sufficient_stats[nb_pics*2:nb_pics*3]  # float64
        stats_background = sufficient_stats[nb_pics*3:]  # float64

        # draw pics/compton proportions
        prior_beta_proportion_pics_compton = 1.
        proportionPicsFond = pl.beta(s1.sum()+prior_beta_proportion_pics_compton,
                                     stats_background.sum()+prior_beta_proportion_pics_compton)

        # draw pics parameters
        emin, emax = offset_energy_canal, offset_energy_canal+kevcan*nb_bins
        lt = time_ns()
        nb_pics, nb_pics_aff = param_update(s1, sx, sx2, _sigmas, offset_variance_energy, coef_variance_energy,
                                            alpha_dirichlet, emin, emax,
                                            pics_weights, pics_locations, pics_variances)
        pics_weights[:nb_pics] *= proportionPicsFond
        now = time_ns()
        t2 += now-lt

        # draw compton with PT
        compton[:] = polya_parallel(stats_background.astype(np.int64), _a_polya, _p_polya,
                                    prior_polya_in, prop_prior_in)
        t3 += time_ns()-now

        compton[...] *= 1-proportionPicsFond
        meanReconstruction+=reconstruction
    meanReconstruction/=nbiter
    print(nbiter, "::", array((t1, t2, t3))*1e-9)
    return array((proportionPicsFond, nb_pics, nb_pics_aff))


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


# draw_labels.parallel_diagnostics(level=4)
# nb.config.THREADING_LAYER = 'threadsafe'
# print(polya_parallel(array([2, 5, 6, 5]), 1., 2., array([1, 1., 1, 1]), 1.))
# print("Threading layer chosen: %s" % nb.threading_layer())
