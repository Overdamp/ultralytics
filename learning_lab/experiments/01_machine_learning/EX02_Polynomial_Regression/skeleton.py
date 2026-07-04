import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.weights = None

    def _get_features(self, X):
        """
        Transform 1D features X to polynomial features up to self.degree
        including a column of ones for the bias/intercept.
        """
        m = len(X)
        # TODO: Initialize X_poly with a column of ones for the bias/intercept
        X_poly = np.ones((m, 1))
        
        # TODO: Add columns containing powers of X up to self.degree
        # Hint: For power in range(1, self.degree + 1), concatenate X ** power horizontally
        for power in range(1, self.degree + 1):
            pass
            
        return X_poly

    def fit(self, X, y):
        """
        Fit polynomial regression model using the Normal Equation:
        w = (X^T @ X)^-1 @ X^T @ y
        """
        # 1. TODO: Extract polynomial features
        X_poly = self._get_features(X)
        
        # 2. TODO: Solve normal equation to compute weights
        # Hint: X_poly.T @ X_poly calculates the matrix multiplication.
        # Hint: Use np.linalg.inv() to compute matrix inverse.
        self.weights = None

    def predict(self, X):
        """
        Predict using the polynomial model.
        """
        # TODO: Compute predictions using X_poly and weights
        # Hint: X_poly @ self.weights
        return None

if __name__ == "__main__":
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0]).reshape(-1, 1)
    # y = 0.5 * x^2 - 2 * x + 3
    y = 0.5 * (X**2) - 2.0 * X + 3.0
    
    model = PolynomialRegression(degree=2)
    model.fit(X, y)
    
    print("--- Training Results ---")
    if model.weights is not None:
        for idx, w in enumerate(model.weights.flatten()):
            print(f"w_{idx} (for X^{idx}): {w:.4f} (expected: {[3.0, -2.0, 0.5][idx]})")
            
        test_X = np.array([[4.0]])
        pred = model.predict(test_X)
        if pred is not None:
            print(f"\nPrediction for X=4.0: {pred[0][0]:.4f} (expected: 3.0000)")
        else:
            print("Prediction logic not implemented.")
    else:
        print("Training weights calculation not implemented.")
