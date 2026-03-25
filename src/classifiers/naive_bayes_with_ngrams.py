#!/usr/bin/env python3
"""
Unified Naive Bayes classifier supporting both char and word n-grams.
Merged from sklearn_nb.py and naive_bayes_categorizer.py
"""
import sys
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Configuration - modify these to change behavior
FEATURE_TYPE = 'char'  # 'char' or 'word'
N_GRAM_SIZE = 3
MAX_FEATURES = 2000

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_and_predict(train_file, test_file, feature_type=FEATURE_TYPE, n=N_GRAM_SIZE, max_features=MAX_FEATURES):
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
    
    # Configure vectorizer based on feature type
    if feature_type == 'char':
        vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(n, n),
            max_features=max_features
        )
    else:  # word
        vectorizer = CountVectorizer(
            analyzer='word',
            ngram_range=(n, n),
            max_features=max_features
        )
    
    # Train vectorizer and transform training data
    X_train = vectorizer.fit_transform(train_texts)
    
    # Train Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_train, train_labels)
    
    # Generate output filename
    test_basename = test_file.rsplit('.', 1)[0]
    # Extract just the filename without path
    test_filename = test_basename.split('/')[-1]
    output_file = f"results/naive_bayes/{test_filename}-pred-naivebayes-{feature_type}-{n}gram-max{max_features}.txt"
    
    # Predict on test data
    with open(test_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            text = line.strip().split('\t')[0]
            X_test = vectorizer.transform([text])
            pred = clf.predict(X_test)[0]
            print(f"{text}\t{pred}")
            f_out.write(f"{text}\t{pred}\n")
    
    print(f"\nPredictions saved to: {output_file}")

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python3 naive_bayes.py train.txt test.txt [feature_type] [n] [max_features]")
        print("  feature_type: 'char' or 'word' (default: char)")
        print("  n: n-gram size (default: 3)")
        print("  max_features: maximum features (default: 2000)")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    # Optional parameters
    feature_type = sys.argv[3] if len(sys.argv) > 3 else FEATURE_TYPE
    n = int(sys.argv[4]) if len(sys.argv) > 4 else N_GRAM_SIZE
    max_features = int(sys.argv[5]) if len(sys.argv) > 5 else MAX_FEATURES
    
    train_and_predict(train_file, test_file, feature_type, n, max_features)

if __name__ == '__main__':
    main()
