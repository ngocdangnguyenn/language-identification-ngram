#!/usr/bin/env python

"""
Script to classify the texts in the dev file.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys
from knn_train import DocCollection, DocVector
from collections import Counter
import pickle

################################################################################

if __name__ == "__main__" : # python way to declare "main" function
  
  # Check if a file was provided as argument, containing preprocessed corpus
  if len(sys.argv) != 3 :
    print("Please provide a training corpus and a preprocessed corpus for prediction!", file=sys.stderr)
    print(f"  Usage: {sys.argv[0]} <train-corpus-file> <dev-corpus-file>", file=sys.stderr)
    exit(-1)  
  training_corpus = sys.argv[1]
  devfilename = sys.argv[2]
  truename = devfilename[:-4]
  model = MultinomialNB()
  
  def extract_data():
    texts = []
    languages = []
    with open(training_corpus, "r", encoding="utf-8") as fic:
      for line in fic :
        whole_data = line.split("\t")
        texts.append(whole_data[0])
        languages.append(whole_data[1])
    return texts, languages
  
  def get_vector(texts):
    vect = CountVectorizer()
    vect.fit(texts)
    return vect
  
    
  texts, languages = extract_data()
  vector = get_vector(texts)
  training_data = vector.transform(texts)
  model = MultinomialNB()
  model.fit(training_data, languages)
  with open(devfilename, "r", encoding="utf-8") as fic, open("results/" + truename + "-pred-naivebayes-bow.txt", "w", encoding="utf-8") as fic_dest:
   for line in fic :
     whole_line = line.split("\t")
     line_text = whole_line[0]
     exploitable_line = vector.transform([line_text])
     fic_dest.write(line_text + "\t" + model.predict(exploitable_line)[0] )
            
