import numpy as np
from collections import Counter
from decision_tree import DecisionTreeScratch

class RandomForestScratch:
    """Improved Random Forest with more trees"""
    
    def __init__(self, n_trees=50, max_depth=40, min_samples_split=3, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        else:
            max_features = n_features
        
        print(f"  Training {self.n_trees} deep decision trees...")
        
        for i in range(self.n_trees):
            if (i + 1) % 10 == 0:
                print(f"    Tree {i + 1}/{self.n_trees}")
            
            X_boot, y_boot = self._bootstrap_sample(X, y)
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            X_boot_subset = X_boot[:, feature_indices]
            
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot_subset, y_boot)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_indices)
        
        return self
    
    def predict(self, X):
        tree_predictions = np.array([
            tree.predict(X[:, feature_indices])
            for tree, feature_indices in zip(self.trees, self.feature_indices)
        ])
        
        predictions = []
        for i in range(X.shape[0]):
            class_counts = Counter(tree_predictions[:, i])
            majority_class = class_counts.most_common(1)[0][0]
            predictions.append(majority_class)
        
        return np.array(predictions)