import numpy as np

class RidgeRegression:
    def __init__(self, lmbda=1.0, degree=2):
        self.lmbda = lmbda
        self.degree = degree
        self.weights = None

    def _get_features(self, X):
        """
        Extract polynomial features up to self.degree, including bias.
        """
        m = len(X)
        X_poly = np.ones((m, 1))
        for power in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** power))
        return X_poly

    def fit(self, X, y):
        """
        Fit Ridge Regression model using:
        w = (X^T @ X + lmbda * I')^-1 @ X^T @ y
        """
        # 1. Transform features
        X_poly = self._get_features(X)
        n_features = X_poly.shape[1]
        
        # 2. TODO: Create a modified identity matrix (I_prime) where the first element is 0.0
        # Hint: np.identity(n_features) creates a standard identity matrix. Set I_prime[0, 0] = 0.0
        I_prime = None
        
        # 3. TODO: Compute Ridge weights using the Normal Equation with L2 penalty
        # Hint: Left side component = X_poly.T @ X_poly + self.lmbda * I_prime
        # Hint: Invert the left side, then multiply by X_poly.T @ y
        self.weights = None

    def predict(self, X):
        """
        Predict using the Ridge model.
        """
        # TODO: Compute predictions using polynomial features and weights
        # Hint: X_poly @ self.weights
        return None

if __name__ == "__main__":
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0]).reshape(-1, 1)
    y = 0.5 * (X**2) - 2.0 * X + 3.0
    
    model = RidgeRegression(lmbda=0.01, degree=2)
    model.fit(X, y)
    
    print("--- Training Results ---")
    if model.weights is not None:
        for idx, w in enumerate(model.weights.flatten()):
            print(f"w_{idx}: {w:.4f}")
            
        test_X = np.array([[4.0]])
        pred = model.predict(test_X)
        if pred is not None:
            print(f"\nPrediction for X=4.0: {pred[0][0]:.4f}")
        else:
            print("Prediction logic not implemented.")
    else:
        print("Training weights calculation not implemented.")
