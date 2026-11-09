import numpy as np

class SVMScratch:
    """Improved SVM with better learning rate and iterations"""
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = {}
        self.biases = {}
        self.classes = []
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        print(f"  Training {len(self.classes)} binary SVMs (1000 iterations)...")
        
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            w = np.zeros(n_features)
            b = 0
            
            for iteration in range(self.n_iterations):
                for idx, x_i in enumerate(X):
                    condition = y_binary[idx] * (np.dot(x_i, w) + b) >= 1
                    
                    if condition:
                        dw = 2 * self.lambda_param * w
                        db = 0
                    else:
                        dw = 2 * self.lambda_param * w - np.dot(x_i, y_binary[idx])
                        db = -y_binary[idx]
                    
                    w -= self.learning_rate * dw
                    b -= self.learning_rate * db
            
            self.weights[cls] = w
            self.biases[cls] = b
        
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            scores[:, i] = np.dot(X, self.weights[cls]) + self.biases[cls]
        
        indices = np.argmax(scores, axis=1)
        return self.classes[indices]