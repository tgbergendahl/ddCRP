import numpy as np
from helper import exch_dirichlet_lhood

def doc_lhood(docs, lambda_):
    docs = np.array(docs)
    if docs.ndim == 1:
        return exch_dirichlet_lhood(docs, lambda_)
    else:
        return exch_dirichlet_lhood(np.sum(docs, axis=0), lambda_)

def doc_lhood_fn(lambda_):
    return lambda dat: doc_lhood(dat, lambda_)

def safe_log(x):
    x = np.maximum(x, 1e-12)
    return np.log(x)

def log_sum(log_vals):
    a = np.max(log_vals)
    return a + np.log(np.sum(np.exp(log_vals - a)))

def heldout_doc_lhood(doc, dists, alpha, eta, post_dir, decay_fn, state):
    log_prior = safe_log(decay_fn(dists))
    log_prior = np.append(log_prior, np.log(alpha))
    log_prior -= log_sum(log_prior)

    # Compute log likelihood of doc under each existing component
    lhoods = []
    for comp in post_dir:
        combined = np.array(comp) + doc
        lhood = exch_dirichlet_lhood(combined, eta) - exch_dirichlet_lhood(comp, eta)
        lhoods.append(lhood)

    # New component: use just the document with symmetric Dirichlet prior
    lhoods.append(exch_dirichlet_lhood(doc, eta))

    log_lhoods = np.array(lhoods)

    # Combine priors and likelihoods
    return log_sum(log_prior + log_lhoods)
