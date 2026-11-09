import numpy as np

class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeScratch:
    """Improved Decision Tree with deeper trees"""
    
    def __init__(self, max_depth=40, min_samples_split=3):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _information_gain(self, y, left_y, right_y):
        parent_gini = self._gini_impurity(y)
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        child_gini = (n_left / n) * self._gini_impurity(left_y) + \
                     (n_right / n) * self._gini_impurity(right_y)
        
        return parent_gini - child_gini
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            if len(thresholds) > 100:
                thresholds = np.random.choice(thresholds, 100, replace=False)
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)
        
        feature_idx, threshold, gain = self._best_split(X, y)
        
        if feature_idx is None or gain <= 0:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(feature_idx, threshold, left, right)
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        y_encoded = np.array([self.class_to_idx[label] for label in y])
        
        print("  Building deep decision tree (max_depth=40)...")
        self.root = self._build_tree(X, y_encoded)
        return self
    
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        predictions = [self._predict_sample(x, self.root) for x in X]
        return np.array([self.idx_to_class[idx] for idx in predictions])