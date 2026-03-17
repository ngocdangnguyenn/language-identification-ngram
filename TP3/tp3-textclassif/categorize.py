#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to classify the texts in the dev file.
"""

import sys
from doc2vec import DocCollection, DocVector
from collections import Counter
import pickle

################################################################################

if __name__ == "__main__" : # python way to declare "main" function
  
  # Check if a file was provided as argument, containing preprocessed corpus
  if len(sys.argv) != 2 :
    print("Please provide a preprocessed corpus for prediction!", file=sys.stderr)
    print(f"  Usage: {sys.argv[0]} <dev-corpus-file>", file=sys.stderr)
    exit(-1)  
  
  # Load the list of vectorized documents from binary file named "model.pkl"    
  docModel = pickle.load(open("model.pkl", 'rb'))
  devfilename = sys.argv[1]   
  
  # Debug: display a few loaded documents to check category and vector
  # for i, doc in enumerate(docModel.docs[:3]):
  #   preview = list(doc.vector.items())[:10]
  #   print(f"DOC {i+1} | cat={doc.category} | vector_sample={preview}")
  
  # Process each document in the dev file
  with open(devfilename, 'r', encoding='UTF-8') as f:
    for line in f:
      line = line.rstrip('\n')
      if not line:
        continue
      text, category = line.split("\t", 1)
      doc = DocVector(text, category)
      predicted_category = docModel.knearest(doc, k=10)
      print(f"{text}\t{predicted_category}")
      
