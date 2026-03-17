#!/usr/bin/env python

"""
Script to classify the texts in the dev file.
"""

import sys
from knn_train import DocVector
from collections import Counter
import pickle
#please comment the right one depending on the model you will provide
#from knn_train import DocCollection  
from knn_train_with_tfidf import DocCollection

################################################################################

if __name__ == "__main__" : # python way to declare "main" function
  
  # Check if a file was provided as argument, containing preprocessed corpus
  if len(sys.argv) != 3 :
    print("Please provide a preprocessed corpus for prediction and a model!", file=sys.stderr)
    print(f"  Usage: {sys.argv[0]} <dev-corpus-file> <model>", file=sys.stderr)
    exit(-1)  
  devfilename = sys.argv[1]      
  modelName = sys.argv[2] 
  docModel = pickle.load(open(modelName,'rb' ))
  modelName = modelName[:-4].split("-")[1:]
  truename = modelName[0]
  suff_supplement = ""
  for particle in modelName[1:] : 
    suff_supplement += particle +"-"
  suff_supplement = suff_supplement[:-1]
with open(devfilename, "r", encoding="utf-8") as fic, open("results/" + truename + "-pred-knearest-" + suff_supplement + ".txt", "w", encoding="utf-8") as fic_dest:
  for file in fic :
    whole_tab = file.split("\t")
    text = whole_tab[0]
    type = whole_tab[1]
    fileVector = DocVector(text, type)
    fic_dest.write(text + "\t" + docModel.knearest(fileVector) + "\n")
        
