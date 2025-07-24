from methods.ddCRP import run_ddCRP
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool


def test_ddCRP(data_path, output_path, alpha_values, beta_values, distance_decay_types, num_iterations):
    """
    Test the ddCRP method with given parameters.
    
    :param data_path: Path to the input data file.
    :param output_path: Path to save the output results.
    :param alpha_values: List of alpha values for the ddCRP method.
    :param beta_values: List of beta values for the ddCRP method.
    :param num_iterations: Number of iterations for the ddCRP method.
    """
    data = pd.read_csv(data_path)
    print(f"Loaded data from {data_path} with shape {data.shape}")

    processes_pool = Pool(6)  # Create a pool of worker processes

    processes_pool.map(run_ddCRP, [(data, output_path, alpha, beta, distance_decay_type, num_iterations)
                                    for alpha in alpha_values
                                    for beta in beta_values
                                    for distance_decay_type in distance_decay_types])

    processes_pool.close()  # Close the pool to new tasks
    processes_pool.join()  # Wait for all worker processes to finish

    print("All ddCRP runs completed.")

    # for alpha in alpha_values:
    #     for beta in beta_values:
    #         for distance_decay_type in distance_decay_types:
    #             print(f"\rRunning ddCRP with alpha={alpha}, beta={beta}, distance decay={distance_decay_type}, iterations={num_iterations}", end="\n", flush=True)
    #             run_ddCRP(data, output_path, alpha, beta, distance_decay_type, num_iterations)
    #             print(f"\rCompleted ddCRP with alpha={alpha}, beta={beta}", end="\n", flush=True)

if __name__ == "__main__":
    device = 'mac'  # Change to 'linux' if running on a Linux machine
    # Set the data and results paths based on the device
    if device == 'linux':
        data_path = "/home/tgb/research/ddCRP/data/gaussian_data.csv"
        results_path = "/home/tgb/research/ddCRP/results"
    else:  # Assuming 'mac'
        data_path = "/Users/tgbergendahl/Research/ddCRP/data/gaussian_data.csv"
        results_path = "/Users/tgbergendahl/Research/ddCRP/results"
    # if the results path does not exist, create it

    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    n = 600
    alpha_values = [0.005, 0.1, 50]
    beta_values = [5, 25, 50, 100]
    distance_decay_types = ['exponential']

    num_iterations = 40

    test_ddCRP(data_path, results_path, alpha_values, beta_values, distance_decay_types, num_iterations)