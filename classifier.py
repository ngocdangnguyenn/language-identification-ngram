#!/usr/bin/env python3
"""
Simple language classifier using character n-grams.
"""
import sys
from collections import Counter, defaultdict

def extract_char_ngrams(text, n=3):
    """Extract character n-grams from text."""
    return [text[i:i+n] for i in range(len(text)-n+1)]

def extract_words_ngrams(text, n=3):
    """Extract words n-grams from text."""
    ngrams = []
    sent = text.split(" ")
    for i in range(len(sent) - n + 1):
        current_ngram = tuple(sent[i:i+n])
        ngrams.append(current_ngram)
    return ngrams

#########Parameters for ngram extraction
ngram_func = extract_words_ngrams    #line to change to choose between characters and words n-grams
n = 3

def train(train_file):
    """Train: learn top n-grams for each language."""
    lang_ngrams = defaultdict(Counter)
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, lang = parts
                ngrams = ngram_func(text, n)
                lang_ngrams[lang].update(ngrams)
    
    # Keep top 100 n-grams per language
    lang_features = {}
    for lang, counter in lang_ngrams.items():
        lang_features[lang] = set([ng for ng, _ in counter.most_common(100)])
    
    return lang_features

def predict(text, lang_features):
    """Predict language by counting matching n-grams."""
    ngrams = set(ngram_func(text, n))
    scores = {}
    
    for lang, features in lang_features.items():
        scores[lang] = len(ngrams & features)
    
    return max(scores, key=scores.get) if scores else 'en'

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 classifier.py train.txt test.txt")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    true_test_filename = test_file[:-4]
    
    # Train
    lang_features = train(train_file)
    # Predict
    with open(test_file, 'r', encoding='utf-8') as f, open(true_test_filename + "-pred1.txt", "w", encoding = "utf-8") as fic_dest:
        for line in f:
            whole_tab = line.strip().split('\t')
            text = whole_tab[0]

            pred_lang = predict(text, lang_features)
            print(f"{text}\t{pred_lang}")
            fic_dest.write(text + "\t" + pred_lang + "\n")
if __name__ == '__main__':
    main()
