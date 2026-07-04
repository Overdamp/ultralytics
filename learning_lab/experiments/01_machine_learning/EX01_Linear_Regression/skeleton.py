import numpy as np

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        """
        Fit linear model to training data X and targets y.
        """
        m = len(X)
        for _ in range(self.epochs):
            # TODO: Compute predictions using y_pred = w * X + b
            y_pred = None
            
            # TODO: Compute error (predictions - ground truth targets)
            error = None
            
            # TODO: Compute gradient of weight (dw) and bias (db)
            # Hint: dw = (1/m) * sum(error * X)
            # Hint: db = (1/m) * sum(error)
            dw = 0.0
            db = 0.0
            
            # TODO: Update w and b using gradient descent update rule
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predict targets using the linear model.
        """
        # TODO: Compute predictions using the learned w and b
        # Hint: return w * X + b
        return None

# Quick test if run directly
if __name__ == "__main__":
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.5 * X + 1.0
    
    model = SimpleLinearRegression(learning_rate=0.05, epochs=1000)
    model.fit(X, y)
    
    print("--- Training Results ---")
    print(f"Learned Weight (w): {model.w:.4f} (expected: 2.5)")
    print(f"Learned Bias (b): {model.b:.4f} (expected: 1.0)")
    
    test_X = np.array([6.0])
    pred = model.predict(test_X)
    if pred is not None:
        print(f"Prediction for X=6.0: {pred[0]:.4f} (expected: 16.0)")
    else:
        print("Prediction logic not implemented yet.")
