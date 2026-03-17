#!/usr/bin/env python

"""
Script to transform a list of documents into a list of word counts (vector)
"""

import sys
import math   
import pickle # useful to dump objects to files and load them back to objects
from collections import Counter # like dict but returns 0 if key absent

################################################################################

class DocVector(object) :
  """
  Represents document with associated vector (Counter dict) and category label (str).
  """
   
  def __init__(self, text, category):
    """
    Creates a vectorised document by counting all words in the `text`.
    """
    self.category = category
    self.normvalue = None
    tokens = text.split()
    self.vector = Counter(tokens)
    
  #########################      

  def norm(self):
    """
    Calculate the norm of the current document vector. Returns a `float`
    """
    if self.normvalue is None:
      result = 0
      for count in self.vector.values():
        result += count ** 2
      self.normvalue = math.sqrt(result)
    return self.normvalue
    
  #########################    
    
  def cosine(self, anotherDoc):
    """
    Cosine similarity of current doc (self) with `anotherDoc`. Returns a `float`
    """
    norm_self = self.norm()
    norm_other = anotherDoc.norm()
    if norm_self == 0 or norm_other == 0:
      return 0.0
    dot = 0
    for token, count in self.vector.items():
      dot += count * anotherDoc.vector.get(token, 0)
    return dot / (norm_self * norm_other)
    

################################################################################

class DocCollection(object):
  """
  Represents a document collection, that is, a list of `DocVector` objects
  """
    
  def __init__(self, filename):
    """
    Read doc collection from `filename`, and initialise list of `DocVector` objects.
    """        
    self.docs = []
    with open(filename, 'r', encoding='UTF-8') as f:
      for line in f:
        line = line.rstrip('\n')
        if not line:
          continue
        text, category = line.split("\t", 1)
        self.docs.append(DocVector(text, category))
      
  def knearest(self, anotherDoc, k=10):
    """
    Performs k-nearest neighbours classification of `anotherDoc`. Returns a `str`
    """
    similarities = []
    for doc in self.docs:
      sim = anotherDoc.cosine(doc)
      similarities.append((sim, doc.category))
    similarities_sorted = sorted(similarities, reverse=True)
    top_k = similarities_sorted[:k]
    category_counts = Counter()
    for sim, cat in top_k:
      category_counts[cat] += 1
    majority_category = category_counts.most_common(1)[0][0]
    return majority_category

################################################################################

if __name__ == "__main__" : # python way to declare "main" function
  
  # Check if a file was provided as argument, containing preprocessed corpus
  if len(sys.argv) != 2 :
    print("Please provide a preprocessed training corpus file!", file=sys.stderr)
    print(f"  Usage: {sys.argv[0]} <train-corpus-file>", file=sys.stderr)
    exit(-1)  
    
  trainfilename = sys.argv[1]    
  # Create document collection from training corpus file
  docCollection = DocCollection(trainfilename) 
  # Save the list of vectorized documents into a binary file named "model.pkl"    
  pickle.dump(docCollection, open("model.pkl", 'wb')) 
  
  
  # Examples from TD 3 : useful to test your code, comment out when ready
  d1 = DocVector("A B C B A", "politique")                 # COMMENT WHEN READY
  d2 = DocVector("C B E B C A B D", "sport")               # COMMENT WHEN READY  
  print(f"||D1||={d1.norm():.2f}  ||D2||={d2.norm():.2f}") # COMMENT WHEN READY
  print(f"cos(D1,D2)={d1.cosine(d2):.2f}")                 # COMMENT WHEN READY


  
  
      
      
