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
        # TODO: Implement soft thresholding logic
        # Hint: If rho > lmbda, return rho - lmbda
        # Hint: If rho < -lmbda, return rho + lmbda
        # Hint: Otherwise, return 0.0
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
                
                # TODO: Compute predictions and residuals without coordinate j
                # Hint: predictions = X @ w_except_j + self.b
                # Hint: residual = y - predictions
                r = None
                
                # TODO: Project residual onto feature X[:, j] (rho_j) and calculate feature norm (z_j)
                # Hint: rho_j = sum(X[:, j] * residual)
                # Hint: z_j = sum(X[:, j] ** 2)
                rho_j = 0.0
                z_j = 1e-8
                
                # TODO: Apply soft thresholding and divide by feature norm to update weight j
                self.w[j] = 0.0

    def predict(self, X):
        """
        Predict using the Lasso model.
        """
        # TODO: Compute predictions using X and learned weights
        # Hint: X @ self.w + self.b
        return None

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = 2.0 * X[:, 0] + np.random.randn(100) * 0.01
    
    model = LassoRegression(lmbda=5.0, epochs=200)
    model.fit(X, y)
    
    print("--- Training Results ---")
    if model.w is not None:
        print(f"w_0 (Predictive): {model.w[0]:.4f} (expected: ~2.0)")
        print(f"w_1 (Noise):      {model.w[1]:.4f} (expected: 0.0000)")
        print(f"Bias (Intercept): {model.b:.4f} (expected: ~0.0)")
        
        test_X = np.array([[0.5, 0.5]])
        pred = model.predict(test_X)
        if pred is not None:
            print(f"\nPrediction for test sample: {pred[0]:.4f}")
        else:
            print("Prediction logic not implemented.")
    else:
        print("Training weights calculation not implemented.")
