import numpy as np
from collections import Counter

class KNeighborsClassifier:
    def __init__(self, K=5):
        self.K = K
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store/memorize training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict classes for a matrix of query points.
        """
        predictions = []
        for x_query in X:
            # Compute Euclidean distances from query point to all training points
            distances = np.sqrt(np.sum((self.X_train - x_query) ** 2, axis=1))
            
            # Find indices of the K nearest neighbors
            k_indices = np.argsort(distances)[:self.K]
            
            # Extract labels of those neighbors
            k_nearest_labels = self.y_train[k_indices].astype(int)
            
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return np.array(predictions)

if __name__ == "__main__":
    X_train = np.array([[1.0, 1.0], [1.5, 2.0], [5.0, 5.0], [6.0, 5.5]])
    y_train = np.array([0, 0, 1, 1])
    
    model = KNeighborsClassifier(K=3)
    model.fit(X_train, y_train)
    
    print("--- Training Results ---")
    test_X = np.array([[1.2, 1.3], [5.5, 5.2]])
    preds = model.predict(test_X)
    print(f"Prediction for point [1.2, 1.3]: Class {preds[0]} (expected: 0)")
    print(f"Prediction for point [5.5, 5.2]: Class {preds[1]} (expected: 1)")
