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
    if denominator == 0:
      return 0.0
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
      self.has_been_factorised = False
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
    list_of_neighbours = sorted(neighbours, key=lambda x: x[0], reverse=True)[:k]
    label_counts = Counter(label for _, label in list_of_neighbours)
    best_count = max(label_counts.values())
    best_labels = [label for label, count in label_counts.items() if count == best_count]
    if len(best_labels) == 1:
      return best_labels[0]

    similarity_sums = Counter()
    for similarity, label in list_of_neighbours:
      if label in best_labels:
        similarity_sums[label] += similarity
    return max(best_labels, key=lambda label: similarity_sums[label])
  
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
      grouped_docs = {}
      for doc_vector in self.documents_vectors:
        if doc_vector.category not in grouped_docs:
          grouped_docs[doc_vector.category] = []
        grouped_docs[doc_vector.category].append(doc_vector)

      new_collection = []
      for documents_with_current_lang in grouped_docs.values():
        new_document = self.concat_texts(documents_with_current_lang)
        new_collection.append(new_document)

      self.documents_vectors = new_collection
      self.has_been_factorised = True
      
      

################################################################################

if __name__ == "__main__" :
  if len(sys.argv) < 2 or len(sys.argv) > 3 :
    print("Please provide a preprocessed training corpus file!", file=sys.stderr)
    print(f"  Usage: {sys.argv[0]} <train-corpus-file> [output-model-file]", file=sys.stderr)
    exit(-1)

  trainfilename = sys.argv[1]
  output_model_file = sys.argv[2] if len(sys.argv) > 2 else None

  docCollection = DocCollection(trainfilename)
  docCollection.fact_colletion()

  if output_model_file:
    output_path = output_model_file
  else:
    suffix = "_gathered" if docCollection.has_been_factorised else ""
    output_path = "models/model" + suffix + ".pkl"

  with open(output_path, 'wb') as model_file:
    pickle.dump(docCollection, model_file)

  print(len(docCollection.documents_vectors))