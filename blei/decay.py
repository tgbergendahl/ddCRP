import numpy as np

def window_decay(w):
    return lambda x: np.array(x) <= w

def logistic_decay(a, b=1):
    return lambda x: 1 / (1 + np.exp((x - a) / b))

def exp_decay(a, b=1.0):
    return lambda x: np.exp(-x / a) * b + 1e-6

def identity_decay(x):
    if np.isscalar(x):
        return 1
    return np.ones_like(x)