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
        X_poly = np.ones((m, 1))
        for power in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** power))
        return X_poly

    def fit(self, X, y):
        """
        Fit polynomial regression model using the Normal Equation:
        w = (X^T @ X)^-1 @ X^T @ y
        """
        X_poly = self._get_features(X)
        XTX = X_poly.T @ X_poly
        self.weights = np.linalg.inv(XTX) @ X_poly.T @ y

    def predict(self, X):
        """
        Predict using the polynomial model.
        """
        X_poly = self._get_features(X)
        return X_poly @ self.weights

if __name__ == "__main__":
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0]).reshape(-1, 1)
    # y = 0.5 * x^2 - 2 * x + 3
    y = 0.5 * (X**2) - 2.0 * X + 3.0
    
    model = PolynomialRegression(degree=2)
    model.fit(X, y)
    
    print("--- Training Results ---")
    for idx, w in enumerate(model.weights.flatten()):
        print(f"w_{idx} (for X^{idx}): {w:.4f} (expected: {[3.0, -2.0, 0.5][idx]})")
        
    test_X = np.array([[4.0]])
    pred = model.predict(test_X)
    print(f"\nPrediction for X=4.0: {pred[0][0]:.4f} (expected: 3.0000)")
