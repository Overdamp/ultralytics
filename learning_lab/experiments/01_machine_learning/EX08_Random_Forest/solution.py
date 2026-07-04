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
        indices = np.random.choice(m, m, replace=True)
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
            # 1. Generate bootstrap sample
            X_boot, y_boot = self._bootstrap_samples(X, y)
            
            # 2. Select random subset of feature indexes
            feat_idx = np.random.choice(n, n_features_to_sample, replace=False)
            self.feat_indices.append(feat_idx)
            
            # 3. Fit decision tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = []
        for tree, feat_idx in zip(self.trees, self.feat_indices):
            preds = tree.predict(X[:, feat_idx])
            tree_preds.append(preds)
            
        tree_preds = np.array(tree_preds).T
        
        # Majority vote
        final_preds = np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])
        return final_preds

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 4)
    # y = 1 if feature 0 > 0.5, else 0 (strong class signal)
    y = (X[:, 0] > 0.5).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=2)
    model.fit(X, y)
    
    print("--- Training Results ---")
    print(f"Number of estimators in forest: {len(model.trees)}")
    
    test_X = np.array([
        [0.8, 0.8, 0.1, 0.1],
        [0.2, 0.2, 0.9, 0.9]
    ])
    preds = model.predict(test_X)
    print(f"\nPrediction for [0.8, 0.8, 0.1, 0.1]: Class {preds[0]} (expected: 1)")
    print(f"Prediction for [0.2, 0.2, 0.9, 0.9]: Class {preds[1]} (expected: 0)")
