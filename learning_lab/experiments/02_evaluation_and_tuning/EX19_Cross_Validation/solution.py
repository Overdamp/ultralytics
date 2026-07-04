import numpy as np

class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        """
        Split dataset into train and validation fold indices.
        """
        m = X.shape[0]
        indices = np.arange(m)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
            
        # Determine fold partition sizes
        fold_sizes = np.full(self.n_splits, m // self.n_splits)
        fold_sizes[:m % self.n_splits] += 1
        
        # Partition indices
        current = 0
        folds = []
        for size in fold_sizes:
            folds.append(indices[current:current + size])
            current += size
            
        # Yield train / val splits
        for i in range(self.n_splits):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            yield train_idx, val_idx

if __name__ == "__main__":
    # Test dataset with 5 samples
    X = np.arange(10).reshape((5, 2))
    
    kf = KFold(n_splits=5, shuffle=False)
    
    print("--- Training Results ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1} | Train: {train_idx} | Val: {val_idx}")
