import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=3, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feat_indices = []

    def _bootstrap_samples(self, X, y):
        m = X.shape[0]
        # TODO: Sample indices randomly with replacement (size m)
        # Hint: Use np.random.choice(m, m, replace=True)
        indices = np.arange(m)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        self.feat_indices = []
        m, n = X.shape
        
        # Determine number of features to sample for each tree
        if self.max_features is None:
            n_features_to_sample = n
        elif self.max_features == 'sqrt':
            n_features_to_sample = int(np.sqrt(n))
        else:
            n_features_to_sample = self.max_features
            
        for _ in range(self.n_estimators):
            # 1. TODO: Generate bootstrap sample (X_boot, y_boot)
            X_boot, y_boot = X, y
            
            # 2. TODO: Select random subset of feature indexes (feat_idx) of size n_features_to_sample
            # Hint: Use np.random.choice(n, n_features_to_sample, replace=False)
            feat_idx = np.arange(n)
            self.feat_indices.append(feat_idx)
            
            # 3. Fit decision tree on subset of features
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = []
        # TODO: Loop over trained trees and feature indexes to gather predictions
        # Hint: Extract features using X[:, feat_idx], predict, and append to tree_preds
        for tree, feat_idx in zip(self.trees, self.feat_indices):
            pass
            
        # TODO: Aggregate predictions using majority voting
        # Hint: Transpose tree_preds matrix and count the most common class for each sample
        final_preds = np.zeros(X.shape[0])
        return final_preds

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 4)
    # y = 1 if feature 0 > 0.5, else 0
    y = (X[:, 0] > 0.5).astype(int)
    
    model = RandomForestClassifier(n_estimators=10, max_depth=5, max_features=2)
    model.fit(X, y)
    
    print("--- Training Results ---")
    if len(model.trees) > 0:
        print(f"Number of estimators in forest: {len(model.trees)}")
        
        test_X = np.array([
            [0.8, 0.8, 0.1, 0.1],
            [0.2, 0.2, 0.9, 0.9]
        ])
        preds = model.predict(test_X)
        if preds is not None and len(preds) > 0:
            print(f"\nPrediction for [0.8, 0.8, 0.1, 0.1]: Class {preds[0]} (expected: 1)")
            print(f"Prediction for [0.2, 0.2, 0.9, 0.9]: Class {preds[1]} (expected: 0)")
        else:
            print("Prediction logic not implemented yet.")
    else:
        print("Model fitting logic not implemented yet.")
