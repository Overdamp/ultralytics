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
        # TODO: Implement Shannon Entropy computation
        # Hint: Count frequency of each class in y using np.bincount(y) or similar.
        # Hint: Sum up -p * log2(p) for each non-zero proportion p.
        return 0.0

    def _information_gain(self, y, X_col, threshold):
        """
        Calculate information gain from splitting on X_col at threshold.
        """
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Split indexes
        left_idx = np.where(X_col <= threshold)[0]
        right_idx = np.where(X_col > threshold)[0]
        
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
            
        # TODO: Compute children weighted average entropy
        # Hint: n_l/n * entropy(left) + n_r/n * entropy(right)
        child_entropy = 0.0
        
        # TODO: Return Information Gain (parent_entropy - child_entropy)
        return 0.0

    def _best_split(self, X, y):
        """
        Determine the best feature index and split threshold.
        """
        best_gain = -1
        split_idx, split_thresh = None, None
        m, n = X.shape
        
        # TODO: Loop through features and unique values to find split maximizing Information Gain
        for feature in range(n):
            X_col = X[:, feature]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                pass
                    
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
            
        # TODO: Split datasets, build left and right children, and return parent node
        # Hint: left_idx = where X[:, split_idx] <= split_thresh
        # Hint: right_idx = where X[:, split_idx] > split_thresh
        left_child = None
        right_child = None
        
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
    if model.root is not None:
        print(f"Root Split Feature: X_{model.root.feature}")
        print(f"Root Split Threshold: {model.root.threshold:.4f}")
        
        test_X = np.array([[11.0, 1.8], [4.0, 0.4]])
        preds = model.predict(test_X)
        if preds is not None and len(preds) > 0 and preds[0] is not None:
            print(f"\nPrediction for [11.0, 1.8]: Class {preds[0]} (expected: 1)")
            print(f"Prediction for [4.0, 0.4]  : Class {preds[1]} (expected: 0)")
        else:
            print("Prediction logic not implemented yet.")
    else:
        print("Model fitting logic not implemented yet.")
