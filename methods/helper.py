import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def gaussian_likelihood(data):
    """
    Calculate the Gaussian log-likelihood of a dataset.
    
    Parameters:
        data (np.ndarray): Data points to calculate the likelihood for.
        
    Returns:
        float: Gaussian log-likelihood of the data.
    """
    n = len(data)
    if n == 0:
        return 1  # Avoid division by zero if no data points
    mean = np.mean(data)
    variance = np.var(data)
    if variance == 0:
        return 1.0  # Avoid division by zero
    likelihood = np.log((1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((data - mean) ** 2) / (2 * variance)))
    return np.exp(np.sum(likelihood) / n)  # Return average log likelihood per point

def lhood_same(alpha):
    """
    Calculate the likelihood a point links to itself based on the alpha parameter.
    Parameters:
        alpha (float): Alpha parameter for the probability calculation.
    Returns:
        float: Log-likelihood of a point linking to itself.
    """
    return alpha

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
    return decay_value

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
    combined_clusters = cluster_1 + cluster_2
    lhood_new = gaussian_likelihood(combined_clusters)
    lhood_c1 = gaussian_likelihood(cluster_1)
    lhood_c2 = gaussian_likelihood(cluster_2)
    total_lhood = decay_value * (lhood_new / (lhood_c1 * lhood_c2))
    return total_lhood

if __name__ == "__main__":
    # Example usage
    x = np.array([1, 2])
    y = np.array([4, 6])

    distance = euclidean_distance(x, y)
    print("Euclidean Distance:", distance)

    exp_decay = ExponentialDecay(alpha=5)
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
    distances = np.linspace(0, 20, 100)
    decay_values = [exp_decay(d) for d in distances]
    plt.plot(distances, decay_values)
    plt.title("Exponential Decay Function")
    plt.xlabel("Distance")
    plt.ylabel("Decay Value")
    plt.grid()
    plt.show()
    print("Exponential decay graph displayed.")

    # test gaussian_likelihood
    data = np.random.normal(loc=0, scale=1, size=100)
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
    cluster_1 = np.random.normal(loc=0, scale=1, size=50)
    cluster_2 = np.random.normal(loc=5, scale=1, size=50)
    prob_new_join = lhood_new_join(x, y, exp_decay, cluster_1, cluster_2)
    print("Probability New Join:", prob_new_join)

    # exponentialize
    print("Exponentialized Probability New Join:", np.exp(prob_new_join))
    print("Exponentialized Probability New No Join:", np.exp(prob_new_no_join))
    print("Exponentialized Probability Same:", np.exp(prob_same))
