#!/usr/bin/env python3
"""
Test langid library on dev.txt
"""
import langid
import sys

if len(sys.argv) != 2:
    print("Usage: python3 test_langid.py dev.txt")
    sys.exit(1)

input_file = sys.argv[1]

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            text = parts[0]
            lang, _ = langid.classify(text)
            print(f"{text}\t{lang}")
