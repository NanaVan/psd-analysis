#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
import math, sys, warnings
import scipy.special as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve

def binom(n, k):
    '''
    effective fraction coefficients for the filter
    n:      half of the order of filter
    k:      terms from the order (-n <= k <= -1, and 1 <= k <= n)
    '''
    return (-1)**(k+1) * math.factorial(n)**2 / math.factorial(n+k) / math.factorial(n-k)

class NONPARAMS_EST(object):
    '''
    non-parameter baseline estimation
    '''
    def __init__(self, data):
        self.data =  data

    def snip(self, method='', order=2, **kwargs):
        snip_method = SNIP(self.data)            
        if method == 'threshold':
            try:
                return snip_method.snip_threshold(order=order, **kwargs) # m, sig  
            except:
                raise ValueError("Invalid value for args in SNIP.snip_threshold method.")
                sys.exit()
        elif method == 'sort':
            try:
                return snip_method.snip_sort(order=order, **kwargs) # m, sort
            except:
                raise ValueError("Invalid value for args in SNIP.snip_sort method.")
                sys.exit()
        elif method == 'auto':
            try:
                return snip_method.snip_auto(order=order, **kwargs) # scale
            except:
                raise ValueError("Invalid value for args in SNIP.snip_auto method.")
                sys.exit()
        else:
            try:
                return snip_method.snip(order=order, **kwargs) # m
            except:
                raise ValueError("Invalid value for args in SNIP.snip method.")
                sys.exit()
        

    def pls(self, method, l, **kwargs):
        pls_method = PLS(self.data)
        if method == 'AsLS':
            try:
                return pls_method.AsLS(l=l, **kwargs) # p, niter
            except:
                raise ValueError("Invalid value for args in PLS.AsLS method.")
                sys.exit()
        elif method == 'airPLS':
            try:
                return pls_method.airPLS(l=l, **kwargs) # niter
            except:
                raise ValueError("Invalid value for args in PLS.airPLS method.")
                sys.exit()
        elif method == 'arPLS':
            try:
                return pls_method.arPLS(l=l, **kwargs) # ratio
            except:
                raise ValueError("Invalid value for args in PLS.arPLS method.")
                sys.exit()
        elif method == 'BrPLS':
            try:
                return pls_method.BrPLS(l=l, **kwargs) # ratio
            except:
                raise ValueError("Invalid value for args in PLS.BrPLS method.")
                sys.exit()
        else:
            raise ValueError("Invalid method for PLS")
            sys.exit()

class SNIP(object):
    '''
    Sensitive Nonlinear Iterative Peak (SNIP) algorithm for 1d spectra
    @ M. Morhac, et al. Appl. Spect. (2008) 91-106
    '''
    def __init__(self, data):
        self.data = data

    def snip(self, m=100, order=2):
        '''
        set the selected points to the minimum value of the calculate range
        original SNIP algorithm
        ---
        m:          parameter for clipping window length
        order:      order of filter (must be even), default 2
        ---
        return
        v:          baseline spectrum
        '''
        if m // 2 < order:
            raise ValueError('window length too small')
        if m % 2 != 0:
            raise ValueError("order must be even")
        v = self.data.copy()
        # prepare binomial coefficients of Pascal's triangle
        order = order//2
        binoms = [[binom(o, k) for k in range(-o, 0)] for o in range(1, order+1)]
        # SNIP procedure
        N = len(v)
        for p in np.arange(0, m, order):
            w = np.minimum(v[p:N-p], np.max(np.vstack(([np.sum([binoms[o][k] * (v[int(k*p/(o+1)):N-2*p+int(k*p/(o+1))] + v[2*p-int(k*p/(o+1)):N-int(k*p/(o+1))]) for k in range(len(binoms[o]))], axis=0) for o in range(order)])), axis=0))
            #w = np.minimum(v[p:N-p], (v[:N-2*p] + v[2*p:])/2) if order == 2 else np.minimum(v[p:N-p], np.maximum((v[:N-2*p] + v[2*p:])/2 , (-v[:N-2*p] + 4*v[int(p/2):N-int(3*p/2)] + 4*v[int(3*p/2):N-int(p/2)] -v[2*p:])/6))
            v[p:N-p] = w
        return v

    def snip_threshold(self, m=100, order=2, sig=1e-3):
        '''
        setting a tolerance for replacing the spectrum
        based on SNIP algorithm
        ---
        m:          parameter for clipping window length
        order:      order of filter (must be even), default 2
        sig:        tolerance between resulting spectrum of each window length
        ---
        return
        v:          baseline spectrum
        '''
        v = self.data.copy()
        # prepare binomial coefficients of Pascal's triangle
        order = order//2
        binoms = [[binom(o, k) for k in range(-o, 0)] for o in range(1, order+1)]
        # SNIP procedure
        N = len(v)
        for p in np.arange(0, m, order):
            w = np.minimum(v[p:N-p], np.max(np.vstack(([np.sum([binoms[o][k] * (v[int(k*p/(o+1)):N-2*p+int(k*p/(o+1))] + v[2*p-int(k*p/(o+1)):N-int(k*p/(o+1))]) for k in range(len(binoms[o]))], axis=0) for o in range(order)])), axis=0))
            #w = np.minimum(v[p:N-p], (v[:N-2*p] + v[2*p:])/2) if order == 2 else np.minimum(v[p:N-p], np.maximum((v[:N-2*p] + v[2*p:])/2 , (-v[:N-2*p] + 4*v[int(p/2):N-int(3*p/2)] + 4*v[int(3*p/2):N-int(p/2)] -v[2*p:])/6))
            #w = np.where(v[p:N-p]/w+w/v[p:N-p]<=2+2*sig, v[p:N-p], w)
            w = np.where(v[p:N-p]/w<=sig, v[p:N-p], w)
            v[p:N-p] = w
        return v

    def snip_sort(self, m=100, order=2, sort=9e-1):
        '''
        replacing the spectrum according to the sorted result
        based on SNIP algorithm
        ---
        m:          parameter for clipping window length
        order:      order of filter (must be even), default 2
        sort:       the precent of the resulting spectrum not to be replaced by the next window length
        ---
        return
        v:          baseline spectrum
        '''
        v = self.data.copy()
        # prepare binomial coefficients of Pascal's triangle
        order = order//2
        binoms = [[binom(o, k) for k in range(-o, 0)] for o in range(1, order+1)]
        # SNIP procedure
        N = len(v)
        for p in np.arange(0, m, order):
            w = np.minimum(v[p:N-p], np.max(np.vstack(([np.sum([binoms[o][k] * (v[int(k*p/(o+1)):N-2*p+int(k*p/(o+1))] + v[2*p-int(k*p/(o+1)):N-int(k*p/(o+1))]) for k in range(len(binoms[o]))], axis=0) for o in range(order)])), axis=0))
            #w = np.minimum(v[p:N-p], (v[:N-2*p] + v[2*p:])/2) if order == 2 else np.minimum(v[p:N-p], np.maximum((v[:N-2*p] + v[2*p:])/2 , (-v[:N-2*p] + 4*v[int(p/2):N-int(3*p/2)] + 4*v[int(3*p/2):N-int(p/2)] -v[2*p:])/6))
            w = np.where((w/v[p:N-p]+v[p:N-p]/w)<=np.sort(w/v[p:N-p]+v[p:N-p]/w)[int(sort*(N-2*p))-1], v[p:N-p], w)
            v[p:N-p] = w
        return v

    def snip_auto(self, order=2, scale=5e-7):
        '''
        auto SNIP algorithm
        finding the suitable m corresponding to fited order
        ---
        order:      order of filter (must be even), default 2
        scale:      scale of the smoothness for m
        ---
        return
        v:          baseline spectrum
                    suitable clipping window length m
        '''
        v = self.data.copy()
        # prepare binomial coefficients of Pascal's triangle
        order = order//2
        binoms = [[binom(o, k) for k in range(-o, 0)] for o in range(1, order+1)]
        # SNIP procedure
        N = len(v)
        p, aver_this = 0, 1
        results = []
        while True:
            v_former = v.copy()
            aver_former = aver_this
            w = np.minimum(v[p:N-p], np.max(np.vstack(([np.sum([binoms[o][k] * (v[int(k*p/(o+1)):N-2*p+int(k*p/(o+1))] + v[2*p-int(k*p/(o+1)):N-int(k*p/(o+1))]) for k in range(len(binoms[o]))], axis=0) for o in range(order)])), axis=0))
            v[p:N-p] = w
            aver_this = np.mean(v_former/v)
            if len(results) == 0:
                if np.abs(aver_this/aver_former-1) <= scale:
                    results.append(v)
            elif len(results) >= 5:
                v = results[2]
                #print("m: {:}".format(p-2*order))
                break
            else:
                if np.abs(aver_this/aver_former-1) <= scale:
                    results.append(v)
                else:
                    results = []
            p += order
        return v, p-2*order

class PLS(object):
    '''
    Reweighted Penalized Least Squares (PLS) for 1d baseline fitting
    starting from 
    @ P. Eilers, H. Boelens. Baseline Correction with Asymmetric Least Squares Smoothing (2005)
    '''
    def __init__(self, data):
        self.data = data

    def AsLS(self, l, p=1e-3, nitermax=50):
        '''
        Asymmetric Least Squares Smoothing (AsLS)
        @ P. Eilers, H. Boelens. Baseline Correction with Asymmetric Least Squares Smoothing (2005) 
        ---
        l:          parameter for smoothness
        p:          parameter for asymmetry, 
                    0.001 <= p <= 0.1 and 102 <= l <= 109 good for a signal with positive peaks
        nitermax:   number of iterations, default to 10
        ---
        return
        z:          baseline spectrum
        '''
        L = len(self.data)
        D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L, L-2))
        D = l * D.dot(D.transpose())
        w = np.ones(L)
        for i in range(nitermax):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + D
            z = spsolve(Z, w*self.data)
            w = p * (self.data > z) + (1 - p) * (self.data <= z)
        return z

    def airPLS(self, l, nitermax=50):
        '''
        Adaptive iterative reweighted penalized least squares for baseline fitting (airPLS)
        @ Z. M. Zhang, S. Chen, Y. Z. Liang. Analyst., 135(2010), 1138-1146
        ---
        l:          parameter for smoothness
        nitermax:   max number of iterations, default to 10
        ---
        return
        z:          baseline spectrum
        '''
        L = len(self.data)
        D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L, L-2))
        D = l * D.dot(D.transpose())
        w = np.ones(L)
        for i in range(nitermax):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + D
            z = spsolve(Z, w*self.data)
            d = self.data - z
            d_abs = np.abs(d[d<0].sum())
            if (d_abs<0.001*np.abs(self.data).sum() or i==nitermax-1):
                if (i==nitermax-1): print('WARING max iteration reached!')
                break
            w = 0. * (self.data >= z) + np.exp((i+1) * d / d_abs) * (self.data < z)
        return z

    def arPLS(self, l, ratio=1e-6, nitermax=50):
        '''
        Asymmetrically reweighted penalized least squares (arPLS)
        @ S. J. Beak, A. Park, Y. J. Ahn, J. Choo. Analyst., 140(2015), 250-257
        ---
        l:          parameter for smoothness
        ratio:      parameter for termination condition ratio 
        nitermax:   max number of iterations, default to 10
        ---
        return
        z:          baseline spectrum
        '''
        L = len(self.data)
        D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L, L-2))
        D = l * D.dot(D.transpose())
        w = np.ones(L)
        for i in range(nitermax):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + D
            z = spsolve(Z, w*self.data)
            d = self.data - z
            d_m, d_sigma = np.mean(d[d<0]), np.std(d[d<0])
            wt = 1. * (self.data <= z) + sp.expit(- 2 * (d - (-d_m + 2 * d_sigma)) / d_sigma) * (self.data > z)
            if np.sqrt(np.sum((w - wt)**2)/ np.sum(w**2)) < ratio: break
            w = wt
        return z

    def BrPLS(self, l, ratio=1e-6, nitermax=50):
        '''
        Bayesian Asymmetrically reweighted penalized least squares (BrPLS)
        @ Q. Wang. Phys. Rev. E. (2022) (to be submitted)
        ---
        l:          parameter for smoothness
        ratio:      parameter for termination condition ratio 
        nitermax:   max number of iterations, default to 10
        ---
        return
        z:          baseline spectrum
        '''
        L, beta = len(self.data), 0.5
        D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L, L-2))
        D = l * D.dot(D.transpose())
        w, z = np.ones(L), self.data.copy()
        warnings.filterwarnings("ignore")
        for i in range(nitermax):
            for i in range(nitermax):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + D
                zt = spsolve(Z, w*self.data)
                d = self.data - zt
                d_m, d_sigma = np.mean(d[d>0]), np.sqrt(np.mean(d[d<0]**2)) 
                w = 1 / (1 + beta / (1 - beta) * np.sqrt(np.pi / 2) * d_sigma / d_m * (1 + sp.erf((d / d_sigma - d_sigma / d_m) / np.sqrt(2))) * np.exp((d / d_sigma - d_sigma / d_m)**2 / 2))
                if np.sqrt(np.sum((z - zt)**2) / np.sum(z**2)) < ratio: break
                z = zt
            if np.abs(beta + np.mean(w) - 1.) < ratio: break
            beta = 1 - np.mean(w)
        return z
