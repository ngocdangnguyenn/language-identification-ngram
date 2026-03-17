#!/usr/bin/env python

"""
Script to classify the texts in the dev file.
"""

import sys
from knn_train import DocCollection, DocVector
import pickle

################################################################################

if __name__ == "__main__" : # python way to declare "main" function
  
  # Check if a file was provided as argument, containing preprocessed corpus
  if len(sys.argv) != 2 :
    print("Please provide a preprocessed corpus for prediction!", file=sys.stderr)
    print(f"  Usage: {sys.argv[0]} <dev-corpus-file>", file=sys.stderr)
    exit(-1)  
  
  # Load the list of vectorized documents from binary file named "model.pkl"  
  use_of_gathered_model = True   #change the model here whether you want to use the full or the factorized model              
  if use_of_gathered_model:
    docModel = pickle.load(open("models/model_gathered.pkl", 'rb'))
    suff_supplement = "-gathered"
  else :
     docModel = pickle.load(open("models/model.pkl", 'rb'))
     suff_supplement = ""
  devfilename = sys.argv[1]      
  truename = devfilename[:-4]

  with open(devfilename, "r", encoding="utf-8") as fic, open("results/" + truename + "-pred-knearest" + suff_supplement + ".txt", "w", encoding="utf-8") as fic_dest:
    for file in fic :
      whole_tab = file.split("\t")
      text = whole_tab[0]
      label = whole_tab[1]
      fileVector = DocVector(text, label)
      fic_dest.write(text + "\t" + docModel.knearest(fileVector) + "\n")
        
