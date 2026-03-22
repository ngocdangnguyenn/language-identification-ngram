#!/usr/bin/env python

"""
Letter Counter Classifier - Naive Baseline
"""

import sys
from collections import Counter

def count_letters(text):
    letters = {}
    for char in text.lower():
        if 'a' <= char <= 'z':
            letters[char] = letters.get(char, 0) + 1
    return letters

def build_language_profiles(filename):
    profiles = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                text, label = parts[0], parts[1]
                if label not in profiles:
                    profiles[label] = Counter()
                profiles[label].update(count_letters(text))
    return profiles

def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    
    all_letters = set(vec1.keys()) | set(vec2.keys())
    dot_product = sum(vec1.get(l, 0) * vec2.get(l, 0) for l in all_letters)
    mag1 = sum(v**2 for v in vec1.values()) ** 0.5
    mag2 = sum(v**2 for v in vec2.values()) ** 0.5
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)

def predict(text, profiles):
    text_letters = count_letters(text)
    best_lang = None
    best_score = -1
    
    for lang, profile in profiles.items():
        score = cosine_similarity(text_letters, dict(profile))
        if score > best_score:
            best_score = score
            best_lang = lang
    
    return best_lang if best_lang else 'en'

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python letter_counter.py <train_file> <test_file>", file=sys.stderr)
        exit(-1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    profiles = build_language_profiles(train_file)
    
    test_name = test_file[:-4]
    with open(test_file, 'r', encoding='utf-8') as f_in, \
         open(f"results/{test_name}-pred-letter-count.txt", 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                text = parts[0]
                predicted_lang = predict(text, profiles)
                f_out.write(text + '\t' + predicted_lang + '\n')
    
    print(f"Predictions saved to results/{test_name}-pred-letter-count.txt")
