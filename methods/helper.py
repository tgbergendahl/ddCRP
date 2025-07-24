import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf

def euclidean_distance(x, y):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
        x (np.ndarray): First point.
        y (np.ndarray): Second point.
        
    Returns:
        float: Euclidean distance between x and y.
    """
    return np.linalg.norm(x - y)

class DistanceDecay:
    """
    Base class for distance decay functions.
    """
    def __call__(self, distance):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def set_param(self, *args, **kwargs):
        """
        Set parameters for the decay function.
        
        Parameters:
            *args: Positional arguments for the decay function.
            **kwargs: Keyword arguments for the decay function.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
class ExponentialDecay(DistanceDecay):
    """
    Exponential decay function.
    
    Parameters:
        alpha (float): Decay rate.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def set_param(self, alpha):
        """
        Set the decay rate.
        
        Parameters:
            alpha (float): New decay rate.
        """
        self.alpha = alpha

    def __call__(self, distance):
        return np.exp(-distance / self.alpha)
    
class WindowDecay(DistanceDecay):
    """
    Window decay function.
    
    Parameters:
        window_size (float): Size of the window.
    """
    def __init__(self, window_size=1.0):
        self.window_size = window_size

    def set_param(self, window_size):
        """
        Set the size of the window.
        
        Parameters:
            window_size (float): New size of the window.
        """
        self.window_size = window_size

    def __call__(self, distance):
        if distance <= self.window_size:
            return 1.0
        else:
            return 0.0
        
class LogisticDecay(DistanceDecay):
    """
    Logistic decay function.
    
    Parameters:
        a (float): Logistic Parameter.
    """
    def __init__(self, alpha=1.0):
        self.a = alpha

    def set_param(self, alpha):
        """
        Set the logistic parameter.
        
        Parameters:
            alpha (float): New logistic parameter.
        """
        self.a = alpha

    def __call__(self, distance):
        return np.exp(-distance + self.a) / (1 + np.exp(-distance + self.a))

def distance_decay(x, y, decay_function):
    """
    Calculate the distance decay between two points using a specified decay function.
    
    Parameters:
        x (np.ndarray): First point.
        y (np.ndarray): Second point.
        decay_function (DistanceDecay): Instance of a DistanceDecay subclass.
        
    Returns:
        float: Decay value based on the distance between x and y.
    """
    distance = euclidean_distance(x, y)
    return decay_function(distance)

# Gibbs Sampling Helper Functions

def gaussian_likelihood(X):
    """
    Calculate the Gaussian log-likelihood of a dataset.
    
    Parameters:
        data (np.ndarray): Data points to calculate the likelihood for.
        
    Returns:
        float: Gaussian log-likelihood of the data.
    """

    if X.shape[0] == 1:
        mu = np.zeros(X.shape[1])
        Sigma = np.eye(X.shape[1])*100  # Use a large covariance for single point
        # print(f"Warning: Single point in cluster, using large covariance matrix {Sigma} for cluster with mean {mu}.")
        return scipy.stats.multivariate_normal.logpdf(X[0], mean=mu, cov=Sigma)
    
    lw = LedoitWolf()
    mu = np.mean(X, axis=0)  # Mean of the data points
    Sigma = lw.fit(X).covariance_  # Use Ledoit-Wolf shrinkage

    log_lhood = scipy.stats.multivariate_normal.logpdf(X, mean=mu, cov=Sigma, allow_singular=True)
    
    total_log_likelihood = np.sum(log_lhood)

    if np.isnan(total_log_likelihood) or np.isinf(total_log_likelihood):
        print(f"Warning: Log-likelihood is NaN or Inf for cluster with mean {mu} and covariance {Sigma}.")
        return -np.inf  # Return negative infinity if log-likelihood is invalid
    return total_log_likelihood

def lhood_same(alpha):
    """
    Calculate the likelihood a point links to itself based on the alpha parameter.
    Parameters:
        alpha (float): Alpha parameter for the probability calculation.
    Returns:
        float: Log-likelihood of a point linking to itself.
    """
    return np.log(alpha)  # Assuming alpha is the probability of linking to itself

def lhood_new_no_join(x, y, decay_function):
    """
    Calculate the likelihood a point links to a new point if this link does not induce new clusters.
    Parameters:
        x (np.ndarray): First point.
        y (np.ndarray): Second point.
        decay_function (DistanceDecay): Instance of a DistanceDecay subclass.
    Returns:
        float: Log-likelihood of a point linking to a new point without inducing new clusters.
    """
    distance = euclidean_distance(x, y)
    decay_value = decay_function(distance)
    return np.log(decay_value)

def lhood_new_join(x, y, decay_function, cluster_1, cluster_2):
    """
    Calculate the likelihood a point links to a new point if this link induces new clusters.
    
    Parameters:
        x (np.ndarray): First point.
        y (np.ndarray): Second point.
        decay_function (DistanceDecay): Instance of a DistanceDecay subclass.
        cluster_1 (ndarray): Cluster 1 data points.
        cluster_2 (ndarray): Cluster 2 data points.
        
    Returns:
        float: Log-likelihood of a point linking to a new point with inducing new clusters.
    """

    distance = euclidean_distance(x, y)
    decay_value = decay_function(distance)
    # f(d_ij) * likelihood of new cluster (cluster 1 + cluster 2) / likelihood of cluster 1 * likelihood of cluster 2
    combined_clusters = np.vstack((cluster_1, cluster_2))
    lhood_new = (gaussian_likelihood(combined_clusters))
    lhood_c1 = (gaussian_likelihood(cluster_1))
    lhood_c2 = (gaussian_likelihood(cluster_2))
    total_lhood = np.log(decay_value) + (lhood_new - (lhood_c1 + lhood_c2))
    return total_lhood

if __name__ == "__main__":
    # Example usage
    x = np.array([1, 2])
    y = np.array([4, 6])

    distance = euclidean_distance(x, y)
    print("Euclidean Distance:", distance)

    exp_decay = ExponentialDecay(alpha=50)
    window_decay = WindowDecay(window_size=10.0)
    logistic_decay = LogisticDecay(alpha=10.0)

    print("Exponential Decay:", distance_decay(x, y, exp_decay))
    print("Window Decay:", distance_decay(x, y, window_decay))
    print("Logistic Decay:", distance_decay(x, y, logistic_decay))

    # # graph logistic decay
    # import matplotlib.pyplot as plt
    # import numpy as np
    # distances = np.linspace(0, 20, 100)
    # decay_values = [logistic_decay(d) for d in distances]
    # plt.plot(distances, decay_values)
    # plt.title("Logistic Decay Function")
    # plt.xlabel("Distance")
    # plt.ylabel("Decay Value")
    # plt.grid()
    # plt.show()
    # print("Logistic decay graph displayed.")

    # graph exponential decay
    distances = np.linspace(0, 100, 1000)
    decay_values = [exp_decay(d) for d in distances]
    plt.plot(distances, decay_values)
    plt.title("Exponential Decay Function")
    plt.xlabel("Distance")
    plt.ylabel("Decay Value")
    plt.grid()
    plt.show()
    print("Exponential decay graph displayed.")

    # test gaussian_likelihood
    data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=1)
    print("Data shape:", data.shape)
    print("Data sample:", data[:5])
    likelihood = gaussian_likelihood(data)
    print("Gaussian Likelihood:", likelihood)
    # test probability_same
    alpha = 0.5
    prob_same = lhood_same(alpha)
    print("Probability Same:", prob_same)
    # test probability_new_no_join
    prob_new_no_join = lhood_new_no_join(x, y, exp_decay)
    print("Probability New No Join:", prob_new_no_join)
    # test probability_new_join
    cluster_1 = np.random.multivariate_normal(mean=[1, 2], cov=[[1, 0], [0, 1]], size=50)
    cluster_2 = np.random.multivariate_normal(mean=[4, 6], cov=[[1, 0], [0, 1]], size=50)
    prob_new_join = lhood_new_join(x, y, exp_decay, cluster_1, cluster_2)
    print("Probability New Join:", prob_new_join)

    # exponentialize
    print("Exponentialized Probability New Join:", np.exp(prob_new_join))
    print("Exponentialized Probability New No Join:", np.exp(prob_new_no_join))
    print("Exponentialized Probability Same:", np.exp(prob_same))
