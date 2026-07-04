import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit Logistic Regression model using Gradient Descent.
        """
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        
        for _ in range(self.epochs):
            # Forward pass: compute predictions
            z = X @ self.w + self.b
            y_pred = self._sigmoid(z)
            
            # Compute gradients
            dw = (1 / m) * (X.T @ (y_pred - y))
            db = np.mean(y_pred - y)
            
            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Predict probability of belonging to class 1.
        """
        z = X @ self.w + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

if __name__ == "__main__":
    X = np.array([[1.0], [2.0], [5.0], [6.0]])
    y = np.array([0, 0, 1, 1])
    
    model = LogisticRegression(learning_rate=0.5, epochs=1000)
    model.fit(X, y)
    
    print("--- Training Results ---")
    print(f"w: {model.w} | b: {model.b:.4f}")
    
    test_X = np.array([[1.5], [5.5]])
    preds = model.predict(test_X)
    probs = model.predict_proba(test_X)
    
    print(f"\nPrediction for X=1.5: Class {preds[0]} (Prob: {probs[0]:.4f})")
    print(f"Prediction for X=5.5: Class {preds[1]} (Prob: {probs[1]:.4f})")
