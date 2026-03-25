#!/usr/bin/env python3
"""
Test langid library on dev.txt
"""
import langid
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python3 test_langid.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
filename = os.path.basename(input_file).replace('.txt', '')

# Create output directory if it doesn't exist
output_dir = "results/langid"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"{filename}-pred-langid.txt")

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            text = parts[0]
            lang, _ = langid.classify(text)
            f_out.write(f"{text}\t{lang}\n")

print(f"Predictions saved to {output_file}")
