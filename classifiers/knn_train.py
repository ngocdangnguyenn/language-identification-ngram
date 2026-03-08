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
   
  def __init__(self, text, category, letter_version = False):
    """
    Creates a vectorised document by counting all words in the `text`.
    """
    self.category = category
    vect = []
    tokenizer = MosesTokenizer()
    text_in_sentences = nltk.tokenize.sent_tokenize(text)
    counter = Counter()
    if not letter_version :
      for sentence in text_in_sentences :
        tokenised_sentence = tokenizer.tokenize(sentence, escape=False)
        counter.update(tokenised_sentence)
    else :
      for sentence in text_in_sentences :
        tokenised_sentence = tokenizer.tokenize(sentence, escape=False)
        for word in tokenised_sentence :
          counter.update(word)
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
    
  def __init__(self, filename, letter_version = False):
    """
    Read doc collection from `filename`, and initialise list of `DocVector` objects.
    """        
    with open(filename, "r", encoding="utf-8") as fic :
      self.documents_vectors = []
      for line in fic :
        processed_line = line.strip().split("\t")
        self.documents_vectors.append(DocVector(processed_line[0], processed_line[1], letter_version))
        self.is_a_letter_model = letter_version
        self.has_been_factorised = False
      
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
  
  ###########Extension : gathering all documents which share the same label in a single document to (try to) be less slow
  def gather_all(self, current_lang):
    result = []
    for doc_vector in self.documents_vectors:
      if doc_vector.category == current_lang : 
        result.append(doc_vector)
    return result
  
  def supress_all(self, tab):
    for doc in tab : 
        self.documents_vectors.remove(doc)

  def concat_texts(self, documents) :
    current_doc= DocVector("", documents[0].category)
    for doc in documents : 
      current_doc.vector += doc.vector
    return current_doc

  def fact_colletion(self):
    """To factorize all DocVectors which share the same label into a big DocVector"""
    if not self.has_been_factorised : 
      new_collection = []
      for doc_vector in self.documents_vectors : 
        current_lang = doc_vector.category
        documents_with_current_lang = self.gather_all(current_lang)
        self.supress_all(documents_with_current_lang)
        new_document = self.concat_texts(documents_with_current_lang)
        new_collection.append(new_document)
      self.documents_vectors = new_collection
      self.has_been_factorised = True
      
      

################################################################################

if __name__ == "__main__" : # python way to declare "main" function
  
  # Check if a file was provided as argument, containing preprocessed corpus
  if len(sys.argv) != 2 :
    print("Please provide a preprocessed training corpus file!", file=sys.stderr)
    print(f"  Usage: {sys.argv[0]} <train-corpus-file>", file=sys.stderr)
    exit(-1)  
    
  trainfilename = sys.argv[1]

  # Create document collection from training corpus file
  docCollection = DocCollection(trainfilename) #you can add True as the end parameter to generate models which counts letters
  docCollection.fact_colletion()  #uncomment this to make a unfactorised model
  if docCollection.has_been_factorised :
    suf_suppl = "_gathered"
  else :
    suf_suppl = ""
  if docCollection.is_a_letter_model : 
    suf_suppl += "_letter"

  # Save the list of vectorized documents into a binary file named "model.pkl"    
  pickle.dump(docCollection, open("models/model" + suf_suppl + ".pkl", 'wb')) 
  print(len(docCollection.documents_vectors))