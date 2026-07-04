import numpy as np

def calculate_bias_variance(y_true, predictions):
    """
    Calculate average bias squared and variance across a set of test points.
    predictions shape: (n_datasets, n_test_points)
    y_true shape: (n_test_points,)
    """
    # 1. Calculate average predictions across datasets: E[f_hat(x)]
    mean_predictions = np.mean(predictions, axis=0)
    
    # 2. Calculate bias squared: (E[f_hat(x)] - f(x))^2 averaged over test points
    bias_squared = np.mean((mean_predictions - y_true) ** 2)
    
    # 3. Calculate variance: E[(f_hat(x) - E[f_hat(x)])^2] averaged over test points
    variance = np.mean(np.var(predictions, axis=0))
    
    return bias_squared, variance

if __name__ == "__main__":
    y_true = np.array([2.0, 3.0, 5.0])
    predictions = np.array([
        [1.8, 3.1, 4.9],
        [2.2, 2.9, 5.1],
        [2.0, 3.0, 5.0]
    ])
    
    bias_sq, var = calculate_bias_variance(y_true, predictions)
    
    print("--- Training Results ---")
    print(f"Bias Squared: {bias_sq:.6f}")
    print(f"Variance    : {var:.6f}")
