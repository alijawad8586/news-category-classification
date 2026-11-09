import numpy as np

class LogisticRegressionScratch:
    """Improved Logistic Regression with better hyperparameters"""
    
    def __init__(self, learning_rate=0.1, n_iterations=1000, regularization=0.001):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = {}
        self.biases = {}
        self.classes = []
        
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        print(f"  Training {len(self.classes)} binary classifiers (1000 iterations)...")
        
        for cls in self.classes:
            y_binary = (y == cls).astype(int)
            w = np.zeros(n_features)
            b = 0
            
            for iteration in range(self.n_iterations):
                z = np.dot(X, w) + b
                predictions = self._sigmoid(z)
                
                error = predictions - y_binary
                dw = (1 / n_samples) * np.dot(X.T, error) + self.regularization * w
                db = (1 / n_samples) * np.sum(error)
                
                w -= self.learning_rate * dw
                b -= self.learning_rate * db
            
            self.weights[cls] = w
            self.biases[cls] = b
        
        return self
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            z = np.dot(X, self.weights[cls]) + self.biases[cls]
            probabilities[:, i] = self._sigmoid(z)
        
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probabilities = probabilities / row_sums
        
        return probabilities
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes[indices]