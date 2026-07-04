import numpy as np

class SupportVectorMachine:
    def __init__(self, C=1.0, learning_rate=0.001, epochs=1000):
        self.C = C
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        """
        Fit SVM using Soft-Margin subgradient descent.
        """
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                # TODO: Check if the sample lies on the correct side of the margin boundary
                # Hint: Calculate y_i * (w^T * x_i + b)
                condition = True
                
                if condition:
                    # TODO: Weight decay regularization update
                    # Hint: self.w -= self.lr * (2 * (1 / self.epochs) * self.w)
                    pass
                else:
                    # TODO: Margin violation update (decay weight and adjust parameters using label and feature vector)
                    # Hint: dw = 2 * (1/epochs) * w - C * y_i * x_i
                    # Hint: db = -C * y_i (which updates bias: b += lr * C * y_i)
                    pass

    def predict(self, X):
        """
        Predict class labels (-1 or +1).
        """
        # TODO: Compute decision function and return predictions using np.sign
        # Hint: np.sign(X @ w + b)
        return None

if __name__ == "__main__":
    X = np.array([[1.0, 2.0], [2.0, 1.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array([-1, -1, 1, 1])
    
    model = SupportVectorMachine(C=10.0, learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    
    print("--- Training Results ---")
    if model.w is not None:
        print(f"w: {model.w} | b: {model.b:.4f}")
        
        test_X = np.array([
            [1.5, 1.5],
            [5.5, 5.5]
        ])
        preds = model.predict(test_X)
        if preds is not None and len(preds) > 0:
            print(f"\nPrediction for [1.5, 1.5]: Class {int(preds[0])} (expected: -1)")
            print(f"Prediction for [5.5, 5.5]: Class {int(preds[1])} (expected: 1)")
        else:
            print("Prediction logic not implemented yet.")
    else:
        print("Model fitting logic not implemented yet.")
