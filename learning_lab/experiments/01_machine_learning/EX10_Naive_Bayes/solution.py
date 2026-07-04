import numpy as np

class GaussianNB:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        """
        Calculate priors, means, and variances for each class.
        """
        self.classes = np.unique(y)
        m = X.shape[0]
        
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / m
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0)

    def _pdf(self, x, mean, var):
        """
        Gaussian Probability Density Function.
        """
        num = np.exp(-((x - mean) ** 2) / (2 * var))
        den = np.sqrt(2 * np.pi * var)
        return num / den

    def _predict_single(self, x):
        posteriors = []
        
        for c in self.classes:
            log_prior = np.log(self.priors[c])
            eps = 1e-15
            pdfs = self._pdf(x, self.means[c], self.vars[c])
            log_likelihood = np.sum(np.log(pdfs + eps))
            
            log_posterior = log_prior + log_likelihood
            posteriors.append((log_posterior, c))
            
        return max(posteriors)[1]

    def predict(self, X):
        """
        Predict classes for query points.
        """
        return np.array([self._predict_single(x) for x in X])

if __name__ == "__main__":
    X_train = np.array([[1.0, 1.0], [1.5, 2.0], [5.0, 5.0], [6.0, 5.5]])
    y_train = np.array([0, 0, 1, 1])
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    print("--- Training Results ---")
    print("Class Priors:", model.priors)
    
    test_X = np.array([
        [1.2, 1.3],
        [5.5, 5.2]
    ])
    preds = model.predict(test_X)
    print(f"\nPrediction for [1.2, 1.3]: Class {preds[0]} (expected: 0)")
    print(f"Prediction for [5.5, 5.2]: Class {preds[1]} (expected: 1)")
