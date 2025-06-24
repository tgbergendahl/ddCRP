from methods.ddCRP import run_ddCRP

def test_ddCRP(data_path, output_path, alpha_values, beta_values, distance_decay_type, num_iterations):
    """
    Test the ddCRP method with given parameters.
    
    :param data_path: Path to the input data file.
    :param output_path: Path to save the output results.
    :param alpha_values: List of alpha values for the ddCRP method.
    :param beta_values: List of beta values for the ddCRP method.
    :param num_iterations: Number of iterations for the ddCRP method.
    """
    for alpha in alpha_values:
        for beta in beta_values:
            print(f"\rRunning ddCRP with alpha={alpha}, beta={beta}, iterations={num_iterations}", end="\n", flush=True)
            run_ddCRP(data_path, output_path, alpha, beta, distance_decay_type, num_iterations)
            print(f"\rCompleted ddCRP with alpha={alpha}, beta={beta}", end="\n", flush=True)

if __name__ == "__main__":
    data_path = "/Users/tgbergendahl/Research/ddCRP/gaussian_data.csv"
    results_path = "/Users/tgbergendahl/Research/ddCRP/results"
    alpha_values = [0.5, 1, 2, 4, 8, 16, 20]
    beta_values = [0.1, 0.5, 1, 2, 5, 10, 20]
    distance_decay_type = "logistic"
    num_iterations = 25
    test_ddCRP(data_path, results_path, alpha_values, beta_values, distance_decay_type, num_iterations)