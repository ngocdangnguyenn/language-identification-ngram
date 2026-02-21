#!/usr/bin/env python3
"""
Test different n-gram sizes (2, 3, 4)
"""
import sys
from collections import Counter, defaultdict

def extract_ngrams(text, n):
    """Extract character n-grams from text."""
    return [text[i:i+n] for i in range(len(text)-n+1)]

def train(train_file, n, top_k):
    """Train: learn top n-grams for each language."""
    lang_ngrams = defaultdict(Counter)
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, lang = parts
                ngrams = extract_ngrams(text, n)
                lang_ngrams[lang].update(ngrams)
    
    # Keep top K n-grams per language
    lang_features = {}
    for lang, counter in lang_ngrams.items():
        lang_features[lang] = set([ng for ng, _ in counter.most_common(top_k)])
    
    return lang_features

def predict(text, lang_features, n):
    """Predict language by counting matching n-grams."""
    ngrams = set(extract_ngrams(text, n))
    scores = {}
    
    for lang, features in lang_features.items():
        scores[lang] = len(ngrams & features)
    
    return max(scores, key=scores.get) if scores else 'en'

def main():
    if len(sys.argv) != 5:
        print("Usage: python3 classifiers/ngram_experimental.py train.txt test.txt n top_k")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    n = int(sys.argv[3])
    top_k = int(sys.argv[4])
    
    # Train
    lang_features = train(train_file, n, top_k)
    
    # Predict
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().split('\t')[0]
            pred_lang = predict(text, lang_features, n)
            print(f"{text}\t{pred_lang}")

if __name__ == '__main__':
    main()
