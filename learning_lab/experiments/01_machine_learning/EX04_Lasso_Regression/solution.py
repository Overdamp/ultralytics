import numpy as np

class LassoRegression:
    def __init__(self, lmbda=1.0, epochs=200):
        self.lmbda = lmbda
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def _soft_threshold(self, rho, lmbda):
        """
        Soft thresholding operator: sign(rho) * max(0, |rho| - lmbda)
        """
        if rho > lmbda:
            return rho - lmbda
        elif rho < -lmbda:
            return rho + lmbda
        else:
            return 0.0

    def fit(self, X, y):
        """
        Fit Lasso using Coordinate Descent.
        """
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        
        for _ in range(self.epochs):
            # Update bias (intercept)
            self.b = np.mean(y - X @ self.w)
            
            # Update each weight coordinate
            for j in range(n):
                # Copy current weights and set coordinate j to 0
                w_except_j = self.w.copy()
                w_except_j[j] = 0.0
                
                # Compute predictions and residuals without coordinate j
                y_pred_except_j = X @ w_except_j + self.b
                r = y - y_pred_except_j
                
                # Project residual onto feature X[:, j]
                rho_j = np.sum(X[:, j] * r)
                z_j = np.sum(X[:, j] ** 2)
                
                # Apply soft thresholding
                self.w[j] = self._soft_threshold(rho_j, self.lmbda) / z_j

    def predict(self, X):
        """
        Predict using the Lasso model.
        """
        return X @ self.w + self.b

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 2)
    # y = 2.0 * X[:, 0] (X[:, 1] is noise)
    y = 2.0 * X[:, 0] + np.random.randn(100) * 0.01
    
    # Regularization lambda = 5.0 (equivalent to alpha = 0.05 in sklearn)
    model = LassoRegression(lmbda=5.0, epochs=200)
    model.fit(X, y)
    
    print("--- Training Results ---")
    print(f"w_0 (Predictive): {model.w[0]:.4f} (expected: ~2.0)")
    print(f"w_1 (Noise):      {model.w[1]:.4f} (expected: 0.0000)")
    print(f"Bias (Intercept): {model.b:.4f} (expected: ~0.0)")
