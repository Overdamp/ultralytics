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
            # TODO: Compute Euclidean distances from query point to all training points
            # Hint: Compute difference vector, square it, sum along rows (axis=1), and take square root
            distances = None
            
            # TODO: Find indices of the K nearest neighbors
            # Hint: Use np.argsort() and slice the first self.K items
            k_indices = []
            
            # TODO: Extract labels of those neighbors from self.y_train
            k_nearest_labels = []
            
            # TODO: Perform majority vote among neighbors
            # Hint: Use Counter(k_nearest_labels).most_common(1) to find the most frequent label
            most_common_label = 0
            predictions.append(most_common_label)
            
        return np.array(predictions)

if __name__ == "__main__":
    X_train = np.array([[1.0, 1.0], [1.5, 2.0], [5.0, 5.0], [6.0, 5.5]])
    y_train = np.array([0, 0, 1, 1])
    
    model = KNeighborsClassifier(K=3)
    model.fit(X_train, y_train)
    
    print("--- Training Results ---")
    if model.X_train is not None:
        test_X = np.array([[1.2, 1.3], [5.5, 5.2]])
        preds = model.predict(test_X)
        if preds is not None and len(preds) > 0 and preds[0] is not None:
            print(f"Prediction for point [1.2, 1.3]: Class {preds[0]} (expected: 0)")
            print(f"Prediction for point [5.5, 5.2]: Class {preds[1]} (expected: 1)")
        else:
            print("Prediction logic not implemented yet.")
    else:
        print("Model fitting logic not implemented yet.")
