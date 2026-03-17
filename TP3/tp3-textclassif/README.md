TP3 - Classification de textes
------------------------------

## Création des vecteurs

Vous devez compléter le script `doc2vec.py` de façon à créer les vecteurs.
Ce script prendra en argument le nom du fichier contenant le corpus pré-traité.
Les corpus fournis `train.txt` et `dev.txt` ont été pré-traités :
  * Encodage UTF-8, LF en fin de ligne,
  * 1 document par ligne, composé de plusieurs phrases,
  * tokénisation avec espaces pour séparer les mots, 
  * casse homogénéisée (tout en minuscules), et
À la fin de chaque ligne, un caractère `TAB` marque la séparation entre le texte
et la catégorie du texte. Les catégories possibles sont `ia`, `cuisine` et
`football`. Le corpus d'évaluation `dev.txt` contient des points d'interrogation
`?` pour indiquer que la catégorie est inconnue et doit être prédite.
  
La sortie de `doc2vec.py` est enregistrée dans un fichier binaire appelé 
`model.pkl`, qui sera ensuite lu par le script de catégorisation.

## Catégorisation des documents

Vous devez compléter le script `doc2vec.py` de façon à classifier tous les 
documents présents dans le corpus d'évaluation `dev.txt`. Ce script écrira la 
sortie sur `stdout`. Pour enregistrer les prédictions, vous pourrez écrire :
```
./categorize.py dev.txt > dev-pred.txt
```

Une fois la prédiction enregistrée dans un fichier, vous pourrez l'évaluer avec
```
./eval.py dev-pred.txt dev-ref.txt
```

Ce script affichera le score d'exactitude (accuracy) de votre système.

## Corpus Wikipédia

La liste de pages appartenant à une catégorie Wikipédia a été obtenue sur le site [https://petscan.wmflabs.org], puis les pages ont été téléchargées à l'aide de `wget`. 
Chaque document de la collection a ensuite été nettoyée pour enlever les balises HTML, 
menus, etc. puis tokénisé à l'aide du tokéniseur Moses français.
