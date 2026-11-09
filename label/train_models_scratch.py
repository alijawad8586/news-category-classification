import pandas as pd
import numpy as np
import re
from collections import Counter
import math
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 NEWS CATEGORY CLASSIFICATION - FROM SCRATCH")
print("=" * 80)

# ============================================================================
# PART 1: TEXT PREPROCESSING & FEATURE EXTRACTION (TF-IDF FROM SCRATCH)
# ============================================================================

class TFIDFVectorizer:
    """TF-IDF Vectorizer from scratch"""
    
    def __init__(self, max_features=5000, min_df=2, max_df=0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = {}
        self.idf = {}
        
    def _tokenize(self, text):
        """Simple tokenization"""
        if pd.isna(text):
            return []
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]
    
    def fit(self, texts):
        """Build vocabulary from training data"""
        print("  Building vocabulary...")
        doc_count = len(texts)
        
        # Count document frequency for each word
        word_doc_freq = Counter()
        all_words = Counter()
        
        for text in texts:
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                word_doc_freq[token] += 1
            all_words.update(tokens)
        
        # Filter by min_df and max_df
        filtered_words = []
        for word, freq in all_words.items():
            doc_freq = word_doc_freq[word]
            doc_freq_ratio = doc_freq / doc_count
            
            if doc_freq >= self.min_df and doc_freq_ratio <= self.max_df:
                filtered_words.append((word, freq))
        
        # Sort by frequency and take top max_features
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        top_words = filtered_words[:self.max_features]
        
        # Build vocabulary
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}
        
        # Calculate IDF
        print("  Calculating IDF...")
        for word in self.vocabulary:
            df = word_doc_freq[word]
            self.idf[word] = math.log((1 + doc_count) / (1 + df)) + 1
        
        print(f"  Vocabulary size: {len(self.vocabulary)}")
        return self
    
    def transform(self, texts):
        """Transform texts to TF-IDF matrix"""
        n_samples = len(texts)
        n_features = len(self.vocabulary)
        
        # Initialize sparse-like representation
        X = np.zeros((n_samples, n_features))
        
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            
            # Calculate term frequency
            tf = Counter(tokens)
            
            # Calculate TF-IDF for each word
            for word, count in tf.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    term_freq = count / len(tokens) if len(tokens) > 0 else 0
                    X[i, idx] = term_freq * self.idf[word]
        
        # Normalize each row
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        X = X / row_norms
        
        return X
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)


# ============================================================================
# PART 2: LOGISTIC REGRESSION FROM SCRATCH
# ============================================================================

class LogisticRegressionScratch:
    """Multi-class Logistic Regression using One-vs-Rest"""
    
    def __init__(self, learning_rate=0.1, n_iterations=1000, regularization=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = {}
        self.biases = {}
        self.classes = []
        
    def _sigmoid(self, z):
        """Sigmoid function with numerical stability"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train using One-vs-Rest approach"""
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        print(f"  Training {len(self.classes)} binary classifiers...")
        
        for cls in self.classes:
            # Binary labels for this class
            y_binary = (y == cls).astype(int)
            
            # Initialize weights
            w = np.zeros(n_features)
            b = 0
            
            # Gradient descent
            for iteration in range(self.n_iterations):
                # Forward pass
                z = np.dot(X, w) + b
                predictions = self._sigmoid(z)
                
                # Calculate gradients
                error = predictions - y_binary
                dw = (1 / n_samples) * np.dot(X.T, error) + self.regularization * w
                db = (1 / n_samples) * np.sum(error)
                
                # Update weights
                w -= self.learning_rate * dw
                b -= self.learning_rate * db
            
            self.weights[cls] = w
            self.biases[cls] = b
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities for each class"""
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            z = np.dot(X, self.weights[cls]) + self.biases[cls]
            probabilities[:, i] = self._sigmoid(z)
        
        # Normalize probabilities
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probabilities = probabilities / row_sums
        
        return probabilities
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes[indices]


# ============================================================================
# PART 3: DECISION TREE FROM SCRATCH
# ============================================================================

class DecisionTreeNode:
    """Node of the decision tree"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Class label for leaf nodes


class DecisionTreeScratch:
    """Decision Tree Classifier from scratch"""
    
    def __init__(self, max_depth=20, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def _gini_impurity(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _information_gain(self, y, left_y, right_y):
        """Calculate information gain"""
        parent_gini = self._gini_impurity(y)
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        child_gini = (n_left / n) * self._gini_impurity(left_y) + \
                     (n_right / n) * self._gini_impurity(right_y)
        
        return parent_gini - child_gini
    
    def _best_split(self, X, y):
        """Find the best split for the data"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            # Sample thresholds if too many
            if len(thresholds) > 50:
                thresholds = np.random.choice(thresholds, 50, replace=False)
            
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
        """Recursively build the tree"""
        n_samples = len(y)
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)
        
        # Find best split
        feature_idx, threshold, gain = self._best_split(X, y)
        
        if feature_idx is None or gain <= 0:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(feature_idx, threshold, left, right)
    
    def fit(self, X, y):
        """Train the decision tree"""
        # Convert labels to integers
        self.classes = np.unique(y)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        y_encoded = np.array([self.class_to_idx[label] for label in y])
        
        print("  Building decision tree...")
        self.root = self._build_tree(X, y_encoded)
        return self
    
    def _predict_sample(self, x, node):
        """Predict a single sample"""
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict class labels"""
        predictions = [self._predict_sample(x, self.root) for x in X]
        return np.array([self.idx_to_class[idx] for idx in predictions])


# ============================================================================
# PART 4: RANDOM FOREST FROM SCRATCH
# ============================================================================

class RandomForestScratch:
    """Random Forest Classifier from scratch"""
    
    def __init__(self, n_trees=50, max_depth=20, min_samples_split=5, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []
        
    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """Train multiple decision trees"""
        n_samples, n_features = X.shape
        
        # Determine number of features to consider
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        else:
            max_features = n_features
        
        print(f"  Training {self.n_trees} decision trees...")
        
        for i in range(self.n_trees):
            if (i + 1) % 10 == 0:
                print(f"    Tree {i + 1}/{self.n_trees}")
            
            # Bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(X, y)
            
            # Random feature subset
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            X_boot_subset = X_boot[:, feature_indices]
            
            # Train decision tree
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot_subset, y_boot)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_indices)
        
        return self
    
    def predict(self, X):
        """Predict using majority voting"""
        # Collect predictions from all trees
        tree_predictions = np.array([
            tree.predict(X[:, feature_indices])
            for tree, feature_indices in zip(self.trees, self.feature_indices)
        ])
        
        # Majority voting
        predictions = []
        for i in range(X.shape[0]):
            class_counts = Counter(tree_predictions[:, i])
            majority_class = class_counts.most_common(1)[0][0]
            predictions.append(majority_class)
        
        return np.array(predictions)


# ============================================================================
# PART 5: SVM FROM SCRATCH (Simplified Linear SVM)
# ============================================================================

class SVMScratch:
    """Linear SVM using One-vs-Rest approach"""
    
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = {}
        self.biases = {}
        self.classes = []
        
    def fit(self, X, y):
        """Train using One-vs-Rest with gradient descent"""
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        print(f"  Training {len(self.classes)} binary SVMs...")
        
        for cls in self.classes:
            # Binary labels: +1 for this class, -1 for others
            y_binary = np.where(y == cls, 1, -1)
            
            # Initialize weights
            w = np.zeros(n_features)
            b = 0
            
            # Gradient descent
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
        """Predict class labels"""
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            scores[:, i] = np.dot(X, self.weights[cls]) + self.biases[cls]
        
        indices = np.argmax(scores, axis=1)
        return self.classes[indices]


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def accuracy_score(y_true, y_pred):
    """Calculate accuracy"""
    return np.mean(y_true == y_pred)

def classification_report_simple(y_true, y_pred):
    """Simple classification report"""
    classes = np.unique(y_true)
    report = {}
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': np.sum(y_true == cls)
        }
    
    return report

# --- MAIN CODE ---
print("\n📂 Loading balanced dataset...")
df = pd.read_csv("balanced_news_data.csv")
print(f"✅ Loaded {len(df):,} rows")

# --- TEXT PREPROCESSING ---
print("\n🔧 Preprocessing text data...")

def clean_text(text):
    """Remove URLs and clean text"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Combine title and content
df['combined_text'] = df.apply(
    lambda row: f"{clean_text(row.get('title', ''))} {clean_text(row.get('content', row.get('content_clean', row.get('text', ''))))}",
    axis=1
)

df = df[df['combined_text'].str.strip() != '']
print(f"✅ After cleaning: {len(df):,} rows")

# --- TRAIN-TEST SPLIT ---
print("\n🔀 Splitting data...")
np.random.seed(42)
indices = np.random.permutation(len(df))
test_size = int(0.2 * len(df))
test_indices = indices[:test_size]
train_indices = indices[test_size:]

X_train_text = df.iloc[train_indices]['combined_text'].values
y_train = df.iloc[train_indices]['category'].values
X_test_text = df.iloc[test_indices]['combined_text'].values
y_test = df.iloc[test_indices]['category'].values

print(f"✅ Training set: {len(X_train_text):,} rows")
print(f"✅ Test set: {len(X_test_text):,} rows")

# --- FEATURE EXTRACTION ---
print("\n📊 Creating TF-IDF features (from scratch)...")
vectorizer = TFIDFVectorizer(max_features=5000, min_df=2, max_df=0.95)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
print(f"✅ Created {X_train.shape[1]:,} features")

# --- DEFINE MODELS ---
print("\n🤖 Initializing models (all from scratch)...")

models = {
    "Logistic Regression (Scratch)": LogisticRegressionScratch(
        learning_rate=0.1,
        n_iterations=500,
        regularization=0.01
    ),
    "Decision Tree (Scratch)": DecisionTreeScratch(
        max_depth=20,
        min_samples_split=5
    ),
    "Random Forest (Scratch)": RandomForestScratch(
        n_trees=30,
        max_depth=20,
        min_samples_split=5
    ),
    "SVM (Scratch)": SVMScratch(
        learning_rate=0.01,
        lambda_param=0.01,
        n_iterations=500
    )
}

# --- TRAIN AND EVALUATE ---
print("\n" + "=" * 80)
print("🎯 TRAINING AND EVALUATING MODELS")
print("=" * 80)

results = {}

for model_name, model in models.items():
    print(f"\n{'─' * 80}")
    print(f"📌 Training: {model_name}")
    print(f"{'─' * 80}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    
    print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
    
    if accuracy >= 0.85:
        print(f"🎉 SUCCESS: Accuracy is ABOVE 85% target!")
    else:
        print(f"⚠️  WARNING: Accuracy is BELOW 85% target")

# --- FINAL COMPARISON ---
print("\n" + "=" * 80)
print("📈 FINAL MODEL COMPARISON")
print("=" * 80)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Model':<35} {'Accuracy':<15} {'Status':<20}")
print("─" * 80)

for model_name, accuracy in sorted_results:
    status = "✅ ABOVE 85%" if accuracy >= 0.85 else "❌ BELOW 85%"
    print(f"{model_name:<35} {accuracy * 100:>6.2f}%{'':<8} {status}")

print("─" * 80)

best_model_name, best_accuracy = sorted_results[0]
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"📊 Accuracy: {best_accuracy * 100:.2f}%")

print("\n" + "=" * 80)
print("✨ TRAINING COMPLETE!")
print("=" * 80)