import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def generate_n_clusters(n_mean=10, n_var=5, seed=42):
    """
    Generate a random number of clusters based on a Gaussian distribution.
    Parameters:
        n_mean (int): Mean number of clusters.
        n_var (int): Variance for the number of clusters.
        seed (int): Random seed for reproducibility.
    Returns:
        int: Randomly generated number of clusters.
    """
    np.random.seed(seed)
    n = np.random.normal(n_mean, n_var)
    n = int(n)
    return n

def generate_gaussian_data(n_clusters=100, n_samples=100, n_features=2, seed=42, mean_scale=25, cov_scale=10):
    """
    Generate synthetic Gaussian data with specified number of clusters, samples, and features.
    Parameters:
        n_clusters (int): Number of clusters to generate.
        n_samples (int): Number of samples per cluster.
        n_features (int): Number of features for each sample.
        seed (int): Random seed for reproducibility.
        mean_scale (float): Scale factor for the means of the clusters.
        cov_scale (float): Scale factor for the covariance matrices.
    Returns:
        np.ndarray: Generated synthetic data of shape (n_clusters * n_samples, n_features).
    """
    np.random.seed(seed)
    data = []
    for _ in range(n_clusters):
        mean = np.random.normal(size=n_features) * mean_scale  # Scale means to avoid overlap
        cov = np.eye(n_features) * np.random.rand() * cov_scale
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        data.append(samples)
    return np.vstack(data)

if __name__ == "__main__":
    n = generate_n_clusters(n_mean=4, n_var=5, seed=42)
    print(f"Number of clusters: {n}")
    data = generate_gaussian_data(n_clusters=n, n_samples=100, n_features=2)
    df = pd.DataFrame(data, columns=[f"feature_{i+1}" for i in range(data.shape[1])])
    df.to_csv("gaussian_data.csv", index=False)
    print("Gaussian data generated and saved to 'gaussian_data.csv'.")
    sns.scatterplot(data=df, x='feature_1', y='feature_2')
    plt.title("Scatter plot of Gaussian data")
    plt.savefig("gaussian_data_plot.png")
    plt.show()
    print("Scatter plot saved as 'gaussian_data_plot.png'.")
    sys.exit(0)