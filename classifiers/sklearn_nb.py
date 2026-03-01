#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifier using scikit-learn with n-grams.
Default: trigrams (n=3) with max 2000 features.
"""
import sys
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Configuration
N_GRAM_SIZE = 3
MAX_FEATURES = 2000

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_and_predict(train_file, test_file):
    """Train Naive Bayes with n-grams and predict."""
    # Load training data
    train_texts = []
    train_labels = []
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                train_texts.append(parts[0])
                train_labels.append(parts[1])
    
    # Vectorize with character n-grams
    vectorizer = CountVectorizer(
        analyzer='char', 
        ngram_range=(N_GRAM_SIZE, N_GRAM_SIZE), 
        max_features=MAX_FEATURES
    )
    X_train = vectorizer.fit_transform(train_texts)
    
    # Train Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_train, train_labels)
    
    # Predict on test data
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().split('\t')[0]
            X_test = vectorizer.transform([text])
            pred = clf.predict(X_test)[0]
            print(f"{text}\t{pred}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 classifiers/sklearn_nb.py train.txt test.txt")
        sys.exit(1)
    
    train_and_predict(sys.argv[1], sys.argv[2])
