from knn_train import DocVector
from knn_train import DocCollection as vanilla_DocCollection
from collections import Counter
from math import *
import sys, pickle


class DocCollection(object):
  """
  Represents a document collection, that is, a list of `DocVector` objects
  """

  def modify_doc_vectors(self):
        """We want to get values in Counters which we can easily modify, so here we will transform Counters into basic dictionnaries"""
        for doc in self.doc_collection.documents_vectors : 
            doc.vector = dict(doc.vector)
    
  def __init__(self, filename, letter_version = False):
    """
    Read doc collection from `filename`, and initialise list of `DocVector` objects.
    """
    def test_all_words_from(doc_vector):
      for word in doc_vector.vector.keys() :
          if word not in self.idf.keys() :
              self.idf[word] = log(corpus_length/(count_how_many_docs_contain(word)))
          doc_vector.vector[word] = doc_vector.vector[word] * self.idf[word]

    def count_how_many_docs_contain(word):
      nb_appereances = 0
      for doc in self.doc_collection.documents_vectors :
          if word in doc.vector.keys():
              nb_appereances += 1
      return nb_appereances
            
    self.has_been_factorised = False
    self.doc_collection = vanilla_DocCollection(filename, letter_version)
    self.modify_doc_vectors()
    self.idf = {}
    self.is_a_letter_model = letter_version
    corpus_length = len(self.doc_collection.documents_vectors)
    for doc in self.doc_collection.documents_vectors : 
       test_all_words_from(doc)


    
      
  def knearest(self, anotherDoc, k=10):
    """
    Performs k-nearest neighbours classification of `anotherDoc`. Returns a `str`
    """
    return self.doc_collection.knearest(anotherDoc, k=10)
  
  ###########Extension : gathering all documents which share the same label in a single document to (try to) be less slow
  def gather_all(self, current_lang):
    return self.doc_collection.gather_all(current_lang)
  
  def supress_all(self, tab):
    self.doc_collection.supress_all(tab)

  def concat_texts(self, documents) :
    self.swap_to_Counter_for_doc_vectors()
    resutl = self.doc_collection.concat_texts(documents)
    self.modify_doc_vectors()
    return resutl
  
  def swap_to_Counter_for_doc_vectors(self):
    for doc in self.doc_collection.documents_vectors : 
        doc.vector = Counter(doc.vector)


  def fact_colletion(self):
    """To factorize all DocVectors which share the same label into a big DocVector"""
    if not self.has_been_factorised : 
      self.doc_collection.fact_colletion()
      self.has_been_factorised = True

if __name__ == "__main__" : # python way to declare "main" function
  
  # Check if a file was provided as argument, containing preprocessed corpus
  if len(sys.argv) != 2 :
    print("Please provide a preprocessed training corpus file!", file=sys.stderr)
    print(f"  Usage: {sys.argv[0]} <train-corpus-file>", file=sys.stderr)
    exit(-1)  
    
  trainfilename = sys.argv[1]
  true_trainfilename = trainfilename[:-4].split("/")[2]

  # Create document collection from training corpus file
  docCollection = DocCollection(trainfilename) #you can add True as the end parameter to generate models which counts letters
  docCollection.fact_colletion()  #(un)comment this to make a unfactorised model
  suf_suppl = "-idf"
  if docCollection.has_been_factorised :
    suf_suppl += "-gathered"
  if docCollection.is_a_letter_model : 
    suf_suppl += "-letter"

  # Save the list of vectorized documents into a binary file named "model.pkl"    
  pickle.dump(docCollection, open("models/model-"+ true_trainfilename + suf_suppl + ".pkl", 'wb')) 
