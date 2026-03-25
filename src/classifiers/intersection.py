#!/usr/bin/env python3
"""
Unified intersection-based classifier supporting both char and word n-grams.
Merged from ngram_simple.py, ngram_experiments.py, and classifier.py
"""
import sys
from collections import Counter, defaultdict

# Configuration - modify these to change behavior
FEATURE_TYPE = 'char'  # 'char' or 'word'
N_GRAM_SIZE = 3
TOP_FEATURES = 100
DEFAULT_LANG = 'en'

def extract_char_ngrams(text, n):
    """Extract character n-grams from text."""
    return [text[i:i+n] for i in range(len(text)-n+1)]

def extract_word_ngrams(text, n):
    """Extract word n-grams from text."""
    words = text.split()
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def get_extractor(feature_type):
    """Get appropriate n-gram extractor based on feature type."""
    if feature_type == 'word':
        return extract_word_ngrams
    else:
        return extract_char_ngrams

def train(train_file, feature_type=FEATURE_TYPE, n=N_GRAM_SIZE, top_k=TOP_FEATURES):
    """Train: learn top n-grams for each language."""
    extractor = get_extractor(feature_type)
    lang_ngrams = defaultdict(Counter)
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, lang = parts
                ngrams = extractor(text, n)
                lang_ngrams[lang].update(ngrams)
    
    # Keep top K n-grams per language
    lang_features = {}
    for lang, counter in lang_ngrams.items():
        lang_features[lang] = set([ng for ng, _ in counter.most_common(top_k)])
    
    return lang_features

def predict(text, lang_features, feature_type=FEATURE_TYPE, n=N_GRAM_SIZE):
    """Predict language by counting matching n-grams."""
    extractor = get_extractor(feature_type)
    ngrams = set(extractor(text, n))
    scores = {}
    
    for lang, features in lang_features.items():
        scores[lang] = len(ngrams & features)
    
    return max(scores, key=scores.get) if scores else DEFAULT_LANG

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python3 intersection.py train.txt test.txt [feature_type] [n] [top_k]")
        print("  feature_type: 'char' or 'word' (default: char)")
        print("  n: n-gram size (default: 3)")
        print("  top_k: number of top features (default: 100)")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    # Optional parameters
    feature_type = sys.argv[3] if len(sys.argv) > 3 else FEATURE_TYPE
    n = int(sys.argv[4]) if len(sys.argv) > 4 else N_GRAM_SIZE
    top_k = int(sys.argv[5]) if len(sys.argv) > 5 else TOP_FEATURES
    
    # Generate output filename
    test_basename = test_file.rsplit('.', 1)[0]
    # Extract just the filename without path
    test_filename = test_basename.split('/')[-1]
    output_file = f"results/intersection/{test_filename}-pred-intersection-{feature_type}-{n}gram-top{top_k}.txt"
    
    # Train
    lang_features = train(train_file, feature_type, n, top_k)
    
    # Predict
    with open(test_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            text = line.strip().split('\t')[0]
            pred_lang = predict(text, lang_features, feature_type, n)
            print(f"{text}\t{pred_lang}")
            f_out.write(f"{text}\t{pred_lang}\n")
    
    print(f"\nPredictions saved to: {output_file}")

if __name__ == '__main__':
    main()
