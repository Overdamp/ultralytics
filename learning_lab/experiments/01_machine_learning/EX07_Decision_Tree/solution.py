import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       # Index of feature to split on
        self.threshold = threshold   # Threshold value for split
        self.left = left             # Left child Node
        self.right = right           # Right child Node
        self.value = value           # Class label if leaf node

    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def _entropy(self, y):
        """
        Calculate Shannon Entropy of labels y.
        """
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _information_gain(self, y, X_col, threshold):
        """
        Calculate information gain from splitting on X_col at threshold.
        """
        parent_entropy = self._entropy(y)
        
        left_idx = np.where(X_col <= threshold)[0]
        right_idx = np.where(X_col > threshold)[0]
        
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
            
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        
        return parent_entropy - child_entropy

    def _best_split(self, X, y):
        """
        Determine the best feature index and split threshold.
        """
        best_gain = -1
        split_idx, split_thresh = None, None
        m, n = X.shape
        
        for feature in range(n):
            X_col = X[:, feature]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                gain = self._information_gain(y, X_col, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = threshold
                    
        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        """
        Recursively construct nodes of the decision tree.
        """
        m, n = X.shape
        n_labels = len(np.unique(y))
        
        # Base cases
        if depth >= self.max_depth or n_labels == 1 or m < 2:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
            
        split_idx, split_thresh = self._best_split(X, y)
        if split_idx is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
            
        left_idx = np.where(X[:, split_idx] <= split_thresh)[0]
        right_idx = np.where(X[:, split_idx] > split_thresh)[0]
        
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        
        return Node(feature=split_idx, threshold=split_thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        """
        Fit the Decision Tree classifier.
        """
        self.root = self._build_tree(X, y)

    def _predict_row(self, node, x):
        if node.is_leaf():
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        """
        Predict labels for input samples X.
        """
        return np.array([self._predict_row(self.root, x) for x in X])

if __name__ == "__main__":
    X = np.array([[10.0, 1.5], [12.0, 2.0], [5.0, 0.5], [6.0, 0.8]])
    y = np.array([1, 1, 0, 0])
    
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)
    
    print("--- Training Results ---")
    print(f"Root Split Feature: X_{model.root.feature}")
    print(f"Root Split Threshold: {model.root.threshold:.4f}")
    
    test_X = np.array([[11.0, 1.8], [4.0, 0.4]])
    preds = model.predict(test_X)
    print(f"\nPrediction for [11.0, 1.8]: Class {preds[0]} (expected: 1)")
    print(f"Prediction for [4.0, 0.4]  : Class {preds[1]} (expected: 0)")
