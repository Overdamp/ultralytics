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
        # TODO: Implement sigmoid equation: 1 / (1 + exp(-z))
        return None

    def fit(self, X, y):
        """
        Fit Logistic Regression model using Gradient Descent.
        """
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        
        for _ in range(self.epochs):
            # TODO: Compute linear output z and predictions y_pred
            # Hint: z = X @ w + b, and pass z through sigmoid
            y_pred = None
            
            # TODO: Compute gradients dw and db
            # Hint: dw = (1/m) * X.T @ (y_pred - y)
            # Hint: db = mean(y_pred - y)
            dw = 0.0
            db = 0.0
            
            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Predict probability of belonging to class 1.
        """
        # TODO: Calculate probabilities by evaluating sigmoid(X @ w + b)
        return None

    def predict(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1).
        """
        # TODO: Get probabilities, and return 1 if probability >= threshold, else 0
        # Hint: (probs >= threshold).astype(int)
        return None

if __name__ == "__main__":
    X = np.array([[1.0], [2.0], [5.0], [6.0]])
    y = np.array([0, 0, 1, 1])
    
    model = LogisticRegression(learning_rate=0.5, epochs=1000)
    model.fit(X, y)
    
    print("--- Training Results ---")
    if model.w is not None:
        print(f"w: {model.w} | b: {model.b:.4f}")
        
        test_X = np.array([[1.5], [5.5]])
        preds = model.predict(test_X)
        probs = model.predict_proba(test_X)
        if preds is not None:
            print(f"\nPrediction for X=1.5: Class {preds[0]} (Prob: {probs[0]:.4f})")
            print(f"Prediction for X=5.5: Class {preds[1]} (Prob: {probs[1]:.4f})")
        else:
            print("Prediction logic not implemented.")
    else:
        print("Training weights calculation not implemented.")
