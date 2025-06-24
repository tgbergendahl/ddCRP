import numpy as np
import scipy.special as sp

def harmonic_mean(x):
    x = np.array(x)
    return -(-np.log(len(x)) + log_sum(-x))

def log_sum(log_x):
    a = np.max(log_x)
    return a + np.log(np.sum(np.exp(log_x - a)))

def exch_dirichlet_lhood(counts, hyper):
    counts = np.array(counts)
    k = len(counts)
    idx = counts > 0
    v = sp.gammaln(k * hyper) - np.sum(idx) * sp.gammaln(hyper)
    v += np.sum(sp.gammaln(counts[idx] + hyper)) - sp.gammaln(np.sum(counts) + k * hyper)
    return v
