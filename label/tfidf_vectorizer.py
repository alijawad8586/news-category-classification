import numpy as np
import re
import pandas as pd
from collections import Counter
import math

class TFIDFVectorizer:
    """Improved TF-IDF Vectorizer with n-grams"""
    
    def __init__(self, max_features=10000, min_df=2, max_df=0.95, ngram_range=(1, 2)):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.vocabulary = {}
        self.idf = {}
        
    def _tokenize(self, text):
        """Tokenize and generate n-grams"""
        if pd.isna(text):
            return []
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        tokens = [t for t in tokens if len(t) > 2]
        
        # Generate n-grams
        ngrams = []
        min_n, max_n = self.ngram_range
        
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
        
        return ngrams
    
    def fit(self, texts):
        """Build vocabulary from training data"""
        print("  Building vocabulary with n-grams...")
        doc_count = len(texts)
        
        word_doc_freq = Counter()
        all_words = Counter()
        
        for text in texts:
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                word_doc_freq[token] += 1
            all_words.update(tokens)
        
        filtered_words = []
        for word, freq in all_words.items():
            doc_freq = word_doc_freq[word]
            doc_freq_ratio = doc_freq / doc_count
            
            if doc_freq >= self.min_df and doc_freq_ratio <= self.max_df:
                filtered_words.append((word, freq))
        
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        top_words = filtered_words[:self.max_features]
        
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}
        
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
        
        X = np.zeros((n_samples, n_features))
        
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            
            for word, count in tf.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    term_freq = count / len(tokens) if len(tokens) > 0 else 0
                    X[i, idx] = term_freq * self.idf[word]
        
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        X = X / row_norms
        
        return X
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)