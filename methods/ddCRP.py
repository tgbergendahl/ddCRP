import pandas as pd
import numpy as np
import os
import sys
from typing import List
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.helper import euclidean_distance, DistanceDecay, ExponentialDecay, WindowDecay, LogisticDecay, lhood_new_no_join, lhood_new_join, lhood_same

class Graph:
    def __init__(self, vertices: int):
        self.vertices = vertices
        self.adjacency_list = [[] for _ in range(vertices)]
#  Use of NoEdge(int, int)
        #prevents duplication of edges
    def add_edge(self, u: int, v: int):
        if self.no_edge(u, v):
            self.adjacency_list[u].append(v)

  
    # Returns true if there does NOT exist
    # any edge from u to v
    def no_edge(self, u: int, v: int):
        return v not in self.adjacency_list[u]

class WCC:
    def __init__(self, directed_graph: Graph):
        self.directed_graph = directed_graph
#  Finds all the connected components
   # of the given undirected graph
    def connected_components(self, undirected_graph: Graph):
        connected_components = []
        is_visited = [False for _ in range(undirected_graph.vertices)]

        for i in range(undirected_graph.vertices):
            if not is_visited[i]:
                component = []
                self.find_connected_component(i, is_visited, component, undirected_graph)
                connected_components.append(component)

        return connected_components
 # Finds a connected component
    # starting from source using DFS
    def find_connected_component(self, src: int, is_visited: List[bool], component: List[int], undirected_graph: Graph):
        is_visited[src] = True
        component.append(src)

        for v in undirected_graph.adjacency_list[src]:
            if not is_visited[v]:
                self.find_connected_component(v, is_visited, component, undirected_graph)
  
    def weakly_connected_components(self):
       #Step 1: Construct the
        # underlying undirected graph
        undirected_graph = Graph(self.directed_graph.vertices)
        for u in range(self.directed_graph.vertices):
            for v in self.directed_graph.adjacency_list[u]:
                undirected_graph.add_edge(u, v)
                undirected_graph.add_edge(v, u)
   # Step 2: Find the connected components
    # of the undirected grap
        return self.connected_components(undirected_graph)
    
class ddCRP_Gibbs:
    def __init__(self, data, distance_decay: DistanceDecay = LogisticDecay(), alpha=1.0, beta=5.0):
        """
        Initialize the ddCRP model.
        
        Parameters:
            data (pd.DataFrame): Input data.
            distance_decay (DistanceDecay): Distance decay function.
            alpha (float): Concentration parameter for the CRP.
            beta (float): Scale parameter for the distance decay.
        """
        self.data = data
        self.distance_decay = distance_decay
        self.alpha = alpha
        self.beta = beta
        self.distance_decay.set_param(beta)  # Set the decay parameter if needed
        # initialize by linking all points to themselves
        self.links = {i: i for i in range(len(data))}
        self.clusters = [[i] for i in range(len(data))]  # Each point starts in its own cluster

    
    def update_clusters(self):
        """
        Update the clusters based on the current links by finding weakly connected components.
        """
        # Create a directed graph from the links
        graph = Graph(len(self.data))
        for i in range(len(self.data)):
            linked_point = self.links[i]
            if linked_point != i:
                graph.add_edge(i, linked_point)
        # Find weakly connected components
        wcc = WCC(graph)
        components = wcc.weakly_connected_components()
        # Update clusters based on the components found
        self.clusters = []
        for component in components:
            if len(component) > 1:
                self.clusters.append(sorted(component))
            else:
                # If a component has only one point, it is isolated
                self.clusters.append([component[0]])
        # Sort clusters by their first element for consistency
        self.clusters.sort(key=lambda x: x[0])
        # Print the updated clusters for debugging
        # if not self.clusters:
        #     print("No clusters found. All points may be isolated.")
        # else:
        #     print(f"Found {len(self.clusters)} clusters.")
        #     for idx, cluster in enumerate(self.clusters):
        #         print(f"Cluster {idx+1}: {cluster}")

        # print(f"Updated clusters: {self.clusters}")

    def get_cluster(self, point_index):
        """
        Get the cluster of a specific point.
        
        Parameters:
            point_index (int): Index of the point.
        
        Returns:
            list: List of indices in the cluster containing the point.
        """
        for cluster in self.clusters:
            if point_index in cluster:
                return cluster
        return []
    
    def sample_assignment(self, x):
        """
        Sample the assignment of point x.
        Parameters:
            x (int): Index of the point to sample.
        """
        # Calculate the likelihood of linking to each point
        lhoods = []
        for j in range(len(self.data)):
            if j == x:
                lhoods.append(lhood_same(self.alpha))
            elif self.get_cluster(x) == self.get_cluster(j):
                lhoods.append(lhood_new_no_join(self.data.iloc[x], self.data.iloc[j], self.distance_decay))
            else:
                x_cluster = self.get_cluster(x)
                # remove x from the cluster to avoid self-linking
                x_cluster_without_x = [i for i in x_cluster if i != x]
                lhoods.append(lhood_new_join(self.data.iloc[x], self.data.iloc[j], self.distance_decay, 
                                             x_cluster_without_x, self.get_cluster(j)))
            # If the likelihood is NaN, we can ignore it or handle it as needed
            if np.isnan(lhoods[-1]):
                lhoods[-1] = -np.inf
            else:
                pass
        # sample from the likelihoods
        lhoods = np.array(lhoods)
        lhoods /= np.sum(lhoods)
        # Sample a new link based on the likelihoods
        # print(f"Sampling new link for point {x} with probabilities: {lhoods}")
        # Ensure the probabilities sum to 1
        if np.sum(lhoods) < 1-1e-5 or np.sum(lhoods) > 1+1e-5:
            print(f"Warning: Likelihoods do not sum to 1 for point {x}. Sum: {np.sum(lhoods)}")

        # graph likelihoods
        # plt.figure(figsize=(10, 5))
        # plt.bar(range(len(self.data)), lhoods, color='blue', alpha=0.7)
        # plt.title(f"Likelihoods for point {x} linking to other points")
        # plt.xlabel("Point Index")
        # plt.ylabel("Probability of Linking")
        # plt.grid(axis='y')
        # plt.show()

        new_link = np.random.choice(range(len(self.data)), p=lhoods)
        # print(f"Point {x} linked to point {new_link}")
        if self.links[x] == new_link:
            # print(f"Point {x} already linked to point {new_link}, no change.")
            return new_link
        self.links[x] = int(new_link)
        # print(f"Updated links: {self.links}")
        self.update_clusters()
        # print("")

    def log_likelihood(self):
        """
        Calculate the log likelihood of the current assignments.
        
        Returns:
            float: Gaussian log likelihood of the current assignments.
        """
        log_likelihood = 0.0

        for cluster in self.clusters:
            points = []
            for i in range(len(cluster)):
                point_index = cluster[i]
                points.append(self.data.iloc[point_index])
            X = np.array(points)
            mu = np.mean(X, axis=0)
            if len(X) < 2:
                continue  # Skip clusters with less than 2 points

            Sigma = np.cov(X, rowvar=False)

            # print(f"Cluster {cluster}:")
            # print(f"Points in cluster: {X}")
            # print(f"Mean (mu): {mu}")
            # print(f"Covariance (Sigma): {Sigma}")

            # Precompute terms
            n, d = X.shape
            det_sigma = np.linalg.det(Sigma)

            if det_sigma <= 0:
                # print(f"Warning: Determinant of covariance matrix is non-positive for cluster {cluster}. Adjusting Sigma.")
                Sigma += 1e-6 * np.eye(Sigma.shape[0])
                det_sigma = np.linalg.det(Sigma)

            inv_sigma = np.linalg.inv(Sigma)

            # Compute Mahalanobis distance for each point
            diff = X - mu
            mahal = np.einsum('ij,jk,ik->i', diff, inv_sigma, diff)

            # Compute log-likelihood
            log_likelihoods = -0.5 * (d * np.log(2 * np.pi) + np.log(det_sigma) + mahal)
            total_log_likelihood = np.sum(log_likelihoods)
            log_likelihood += total_log_likelihood

        return log_likelihood
    
    def run_gibbs_sampling(self, iterations=1):
        """
        Run the Gibbs sampling algorithm.
        
        Parameters:
            iterations (int): Number of iterations to run the Gibbs sampler.
        """
        log_likelihoods = []
        for i in range(iterations):
            for x in range(len(self.data)):
                new_link = self.sample_assignment(x)
            lhood = self.log_likelihood()
            log_likelihoods.append(lhood)
            print(f"\rIteration {i+1}/{iterations} completed. Number of clusters: {len(self.clusters)}", end="", flush=True)
        # After sampling, links will contain the final assignments
        print("\n")
        return log_likelihoods

def run_ddCRP(data, output_path, alpha, beta, distance_decay_type='logistic', iterations=10):
    """
    Run the ddCRP Gibbs sampler on the provided data.
    
    Parameters:
        data (pd.DataFrame): Input data containing features.
        alpha (float): Concentration parameter for the CRP.
        beta (float): Scale parameter for the distance decay.
        distance_decay_type (str): Type of distance decay function to use ('logistic', 'exponential', 'window').
        iterations (int): Number of Gibbs sampling iterations.
    """

    # Ensure output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Initialize distance decay function
    if distance_decay_type == 'logistic':
        distance_decay = LogisticDecay(alpha=beta)
    elif distance_decay_type == 'exponential':
        distance_decay = ExponentialDecay(alpha=beta)
    elif distance_decay_type == 'window':
        distance_decay = WindowDecay(window_size=beta)
    else:
        raise ValueError("Invalid distance decay type. Choose from 'logistic', 'exponential', or 'window'.")
    
    # Initialize ddCRP model
    model = ddCRP_Gibbs(data, distance_decay=distance_decay, alpha=alpha, beta=beta)
    # print("ddCRP Gibbs sampler model initialized.")
    
    # Run Gibbs sampling
    lhoods = model.run_gibbs_sampling(iterations=iterations)
    # print("Gibbs sampling completed.")

    clusters = model.clusters

    # Plot results and save to output path
    plt.figure(figsize=(18, 10))
    colors = plt.cm.get_cmap('hsv', len(clusters))
    for idx, cluster in enumerate(clusters):
        cluster_points = data.iloc[cluster]
        plt.scatter(cluster_points['feature_1'], cluster_points['feature_2'], color=colors(idx), label=f'Cluster {idx+1}', s=10)
    plt.title(f"Clusters after Gibbs Sampling, Alpha={alpha}, Beta={beta}, Distance Decay={distance_decay_type}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.savefig(output_path+f"/clusters_alpha{alpha}_beta{beta}_{distance_decay_type}.png")
    # close figure
    plt.close()
    # print(f"Clusters plot saved to {output_path}/clusters_alpha{alpha}_beta{beta}_{distance_decay_type}.png")

    # Plot log likelihoods
    plt.figure(figsize=(10, 5))
    plt.xticks(range(1, iterations + 1))
    plt.xlim(1, iterations)
    plt.ylim(min(lhoods), max(lhoods))
    plt.plot(range(1, iterations + 1), lhoods, marker='o', linestyle='-', color='blue')
    plt.title(f"Log Likelihoods over Iterations, Alpha={alpha}, Beta={beta}, Distance Decay={distance_decay_type}")
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.grid()
    plt.savefig(output_path+f"/log_likelihoods_alpha{alpha}_beta{beta}_{distance_decay_type}.png")
    # close figure
    plt.close()
    # print(f"Log likelihoods plot saved to {output_path}/log_likelihoods_alpha{alpha}_beta{beta}_{distance_decay_type}.png")
    
    # Save log likelihoods to a CSV file
    lhoods_df = pd.DataFrame({'Iteration': range(1, iterations + 1), 'Log Likelihood': lhoods})
    lhoods_df.to_csv(output_path+f"/log_likelihoods_alpha{alpha}_beta{beta}_{distance_decay_type}.csv", index=False)
    # print(f"Log likelihoods saved to {output_path}/log_likelihoods_alpha{alpha}_beta{beta}_{distance_decay_type}.csv")

    return

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("/Users/tgbergendahl/Research/ddCRP/gaussian_data.csv")
    print(f"{len(data)} points loaded successfully.")

    model = ddCRP_Gibbs(data, distance_decay=ExponentialDecay(alpha=1), alpha=6.0, beta=1)
    print("ddCRP Gibbs sampler model initialized.")

    # lhood = model.log_likelihood()
    # print(f"Initial log likelihood: {lhood}")

    clusters = model.run_gibbs_sampling(iterations=10)
    print(f"Clusters after sampling: {clusters}")
    # plot points with colors based on clusters
    colors = plt.cm.get_cmap('hsv', len(clusters))
    for idx, cluster in enumerate(clusters):
        cluster_points = data.iloc[cluster]
        plt.scatter(cluster_points['feature_1'], cluster_points['feature_2'], color=colors(idx), label=f'Cluster {idx+1}', s=10)
    plt.title("Clusters after Gibbs Sampling")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.savefig("clusters_after_sampling.png")
    plt.show()
    print("Final links:", model.links)
    print("Gibbs sampling completed.")
