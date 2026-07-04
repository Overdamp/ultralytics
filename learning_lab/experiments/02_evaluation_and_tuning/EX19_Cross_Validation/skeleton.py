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
        
        # TODO: Shuffle indices if self.shuffle is True
        # Hint: Use np.random.shuffle(indices) after setting np.random.seed if random_state is not None
        
        # TODO: Calculate fold sizes and partition indices into a list of n_splits sub-arrays
        # Hint: Use np.full(n_splits, m // n_splits) and add remainder indices
        folds = []
        
        # TODO: Loop over the splits and yield train/val index pairs
        # Hint: For split i, val_idx is folds[i], train_idx is the concatenation of other folds
        for i in range(self.n_splits):
            val_idx = np.array([])
            train_idx = np.array([])
            yield train_idx, val_idx

if __name__ == "__main__":
    X = np.arange(10).reshape((5, 2))
    
    kf = KFold(n_splits=5, shuffle=False)
    
    print("--- Training Results ---")
    splits_count = 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        splits_count += 1
        if len(train_idx) > 0 or len(val_idx) > 0:
            print(f"Fold {fold + 1} | Train: {train_idx} | Val: {val_idx}")
            
    if splits_count == 0 or (len(train_idx) == 0 and len(val_idx) == 0):
        print("K-Fold partition logic not implemented yet.")
