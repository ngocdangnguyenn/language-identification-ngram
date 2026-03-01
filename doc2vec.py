#!/usr/bin/env python

"""
Script to transform a list of documents into a list of word counts (vector)
"""
import nltk
import numpy as np
from sacremoses import MosesTokenizer
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
    vect = []
    tokenizer = MosesTokenizer()
    text_in_sentences = nltk.tokenize.sent_tokenize(text)
    counter = Counter()
    for sentence in text_in_sentences :
      tokenised_sentence = tokenizer.tokenize(sentence, escape=False)
      counter.update(tokenised_sentence)
    self.vector = counter
    self.normvalue = None



    
  #########################      

  def norm(self):
    """
    Calculate the norm of the current document vector. Returns a `float`
    """
    if (self.normvalue == None) : 
      work_array = np.array(list(self.vector.values()))
      work_array = work_array ** 2
      self.normvalue = math.sqrt(np.sum(work_array))
    return self.normvalue
    
  #########################    
    
  def cosine(self, anotherDoc):
    """
    Cosine similarity of current doc (self) with `anotherDoc`. Returns a `float`
    """
    numerator = 0
    for key in self.vector.keys() :
        numerator += self.vector[key] * anotherDoc.vector[key]
    denominator = self.norm() * anotherDoc.norm()
    return numerator / denominator
    

################################################################################

class DocCollection(object):
  """
  Represents a document collection, that is, a list of `DocVector` objects
  """
    
  def __init__(self, filename):
    """
    Read doc collection from `filename`, and initialise list of `DocVector` objects.
    """        
    with open(filename, "r", encoding="utf-8") as fic :
      self.documents_vectors = []
      for line in fic :
        processed_line = line.strip().split("\t")
        self.documents_vectors.append(DocVector(processed_line[0], processed_line[1]))
      
  def knearest(self, anotherDoc, k=10):
    """
    Performs k-nearest neighbours classification of `anotherDoc`. Returns a `str`
    """
    neighbours = []
    for document in self.documents_vectors : 
      similarity = document.cosine(anotherDoc)
      category = document.category
      neighbours.append((similarity, category))
    list_of_neighbours = sorted(neighbours, reverse=True)[:k]
    counter = Counter()
    counter.update(list_of_neighbours)
    return counter.most_common(1)[0][0][1]

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
