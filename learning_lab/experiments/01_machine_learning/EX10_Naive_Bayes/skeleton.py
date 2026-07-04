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
        
        # TODO: Compute priors, means, and variances of features for each class c
        for c in self.classes:
            X_c = X[y == c]
            # Prior P(c)
            self.priors[c] = X_c.shape[0] / m
            # Mean and variance along features
            self.means[c] = np.zeros(X.shape[1])
            self.vars[c] = np.ones(X.shape[1])

    def _pdf(self, x, mean, var):
        """
        Gaussian Probability Density Function.
        """
        # TODO: Implement normal distribution probability density equation
        # Hint: 1 / sqrt(2 * pi * var) * exp(-(x - mean)^2 / (2 * var))
        return np.ones_like(x)

    def _predict_single(self, x):
        posteriors = []
        
        # TODO: Compute log-posterior probability for each class, and select class with maximum
        for c in self.classes:
            # Start with log of prior
            log_prior = np.log(self.priors[c])
            
            # Sum up log of PDF densities for features
            # Hint: Sum up log(pdf + eps)
            log_likelihood = 0.0
            
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
    if model.classes is not None:
        print("Class Priors:", model.priors)
        
        test_X = np.array([
            [1.2, 1.3],
            [5.5, 5.2]
        ])
        preds = model.predict(test_X)
        if preds is not None and len(preds) > 0:
            print(f"\nPrediction for [1.2, 1.3]: Class {preds[0]} (expected: 0)")
            print(f"Prediction for [5.5, 5.2]: Class {preds[1]} (expected: 1)")
        else:
            print("Prediction logic not implemented yet.")
    else:
        print("Model fitting logic not implemented yet.")
