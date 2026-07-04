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
        X_poly = self._get_features(X)
        n_features = X_poly.shape[1]
        
        # Modify identity matrix to not penalize intercept (w_0)
        I_prime = np.identity(n_features)
        I_prime[0, 0] = 0.0
        
        XTX = X_poly.T @ X_poly
        left_side = XTX + self.lmbda * I_prime
        self.weights = np.linalg.inv(left_side) @ X_poly.T @ y

    def predict(self, X):
        """
        Predict using the Ridge model.
        """
        X_poly = self._get_features(X)
        return X_poly @ self.weights

if __name__ == "__main__":
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0]).reshape(-1, 1)
    y = 0.5 * (X**2) - 2.0 * X + 3.0
    
    # Run with small regularization lambda=0.01
    model = RidgeRegression(lmbda=0.01, degree=2)
    model.fit(X, y)
    
    print("--- Training Results ---")
    for idx, w in enumerate(model.weights.flatten()):
        print(f"w_{idx}: {w:.4f}")
        
    test_X = np.array([[4.0]])
    pred = model.predict(test_X)
    print(f"\nPrediction for X=4.0: {pred[0][0]:.4f}")
