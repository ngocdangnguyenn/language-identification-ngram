# Language Identification with Classical NLP (L3 TAL Project)
### Auteurs
- ACHOURI Doria
- NGUYEN Ngoc Dang Nguyen

Ce projet a été réalisé dans le cadre du cours **TAL (L3)** à Aix-Marseille Université. Il traite une tâche de classification supervisée : identifier automatiquement la langue d’un texte court à partir d’exemples annotés.

Le corpus est extrait d’Europarl, qui contient des transcriptions multilingues de débats du Parlement européen. Le choix méthodologique est volontairement centré sur des approches classiques, légères et interprétables, afin de comparer des modèles reproductibles sans recourir au deep learning.

---

## Objectif

L’objectif est de prédire la langue d’un texte parmi 21 catégories ISO (`bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`, `fi`, `fr`, `hu`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `sk`, `sl`, `sv`).

Le système est entraîné sur `train.txt`, comparé sur `dev.txt`, puis appliqué à `test.txt` où les labels sont remplacés par `??`. Le format de sortie attendu est identique au format d’entrée : texte original + tabulation + catégorie prédite.

L’évaluation est réalisée avec l’accuracy via `eval.py`, ce qui permet de mesurer simplement la proportion de prédictions correctes.

---

## Méthodes implémentées

1. **Baseline aléatoire**  
Cette méthode assigne une langue au hasard parmi les 21 classes possibles. Elle ne fait aucun apprentissage, mais elle sert de repère minimal pour vérifier que les autres systèmes apportent un vrai gain.

2. **Intersection de n-grammes**  
On extrait des n-grammes (de caractères ou de mots) fréquents pour chaque langue dans `train.txt`, puis on choisit la langue qui a l’intersection la plus forte avec le texte à classer. L’approche est simple et lisible, mais sa qualité dépend fortement du choix des paramètres (`n`, `top_k`).

3. **k plus proches voisins (kNN)**  
Chaque texte est représenté sous forme de vecteur de fréquences avec un comptage de mots ou de lettres, puis comparé aux exemples d’entraînement avec une similarité cosinus. La classe est décidée à partir des voisins les plus proches. Cette méthode est une bonne référence classique, mais elle est plus coûteuse en prédiction. Une variante avec pondération TF-IDF est également testée.

4. **Naive Bayes**  
Deux variantes sont implémentées avec `scikit-learn` :
- une version simple en bag-of-words (`naive_bayes_simple.py`) avec `CountVectorizer` par défaut ;
- une version à base de n-grammes de caractères ou de mots (`naive_bayes_with_ngrams.py`).  
En pratique, la version caractères n-grammes donne les meilleurs résultats sur ce jeu de données.

5. **Régression logistique**  
Une régression logistique multinomiale est entraînée sur des vecteurs bag-of-words (`CountVectorizer`). Elle permet de comparer un autre classifieur linéaire simple aux variantes de Naive Bayes.

6. **Comparaison externe (`langid`)**  
On teste aussi `langid`, une bibliothèque externe prête à l’emploi, pour disposer d’un point de comparaison supplémentaire. Cela permet de situer le niveau du prototype, même si ce type d’outil est moins contrôlable et pas toujours optimisé pour le corpus du projet.

---

## Structure du projet

```text
.
├── baseline.py
├── eval.py
├── train.txt
├── dev.txt
├── test.txt
├── classifiers/
│   ├── intersection.py
│   ├── knn_train.py
│   ├── knn_predict.py
│   ├── knn_train_with_tfidf.py
│   ├── naive_bayes_simple.py
│   ├── naive_bayes_with_ngrams.py
│   └── logistic_regression.py
├── models/
├── results/
├── tools/
│   └── test_langid.py
└── CONSIGNES_SUJET.md
```

---

## Installation

### Dépendances

```bash
pip install numpy nltk sacremoses scikit-learn langid
```

Téléchargement NLTK (première exécution) :

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## Format des données

Chaque ligne suit le format :

```text
<texte>\t<label>
```

- `train.txt` / `dev.txt` : labels connus
- `test.txt` : label `??` à remplacer par la prédiction

---

## Exécution rapide

Toutes les commandes se lancent depuis la racine du projet.

### Baseline

```bash
python baseline.py dev.txt > results/dev-pred-baseline.txt
python eval.py results/dev-pred-baseline.txt dev.txt
```

### Intersection (char 3-gram, top 100)

```bash
python classifiers/intersection.py train.txt dev.txt char 3 100
python eval.py results/dev-pred-intersection-char-3gram-top100.txt dev.txt
```

### Naive Bayes simple (bag-of-words)

```bash
python classifiers/naive_bayes_simple.py train.txt dev.txt
python eval.py results/dev-pred-naivebayes-bow.txt dev.txt
```

### Naive Bayes n-grammes (char 3-gram, 2000 features)

```bash
python classifiers/naive_bayes_with_ngrams.py train.txt dev.txt char 3 2000
python eval.py results/dev-pred-naivebayes-char-3gram-max2000.txt dev.txt
```

### kNN
Décommenter l'import dans knn_predict.py: from knn_train import DocCollection  
Commenter l'import dans knn_predict.py : from knn_train_with_tfidf import DocCollection  

Il y a également des options pour factoriser le modèle et aussi simplement compter les lettres dans le texte (voir commentaires à la fin de knn_train pour activer ces options)
```bash
python classifiers/knn_train.py train.txt
python classifiers/knn_predict.py dev.txt models/<nom du modèle à utiliser>
python eval.py results/dev-pred-knearest-gathered.txt dev.txt
```

### kNN avec TF-IDF
Commenter l'import dans knn_predict.py: from knn_train import DocCollection  
Décommenter l'import dans knn_predict.py : from knn_train_with_tfidf import DocCollection  
```bash
python classifiers/knn_train_with_tfidf.py train.txt
python classifiers/knn_predict.py dev.txt models/<nom du modèle à utiliser>
python eval.py results/dev-pred-knearest-idf-gathered.txt dev.txt
```

### Régression logistique (bag-of-words)

```bash
python classifiers/logistic_regression.py train.txt dev.txt
python eval.py results/dev-pred-logreg.txt dev.txt
```

### langid (comparaison)

```bash
python tools/test_langid.py dev.txt > results/dev-pred-langid.txt
python eval.py results/dev-pred-langid.txt dev.txt
```

---

## Évaluation

Le script [eval.py](eval.py) calcule l’accuracy entre un fichier de prédiction et un fichier de référence :

```bash
python eval.py <fichier_prediction> <fichier_reference>
```

Exemple :

```bash
python eval.py results/dev-pred-naivebayes-char-3gram-max2000.txt dev.txt
```

---

## Générer la prédiction finale pour test.txt

Point important pour la remise : **il faut un fichier de sortie**, pas seulement un affichage terminal.

Exemple avec Naive Bayes n-grammes :

```bash
python classifiers/naive_bayes_with_ngrams.py train.txt test.txt char 3 2000
```

Fichier généré :

- `results/test-pred-naivebayes-char-3gram-max2000.txt`

Ce fichier contient la même structure que `test.txt`, avec `??` remplacé par les labels prédits.

Les fichiers générés ont un nom qui permet normalement d'aisément retrouver la source et la méthode utilisée pour les générer. Ils sont dans le fichier results pour la plupart, sauf les modèles qui seront générés dans le fichier models.

---

## Résultats obtenus

Mesures observées sur `dev.txt` :

| Méthode | Fichier de prédiction | Accuracy |
|---|---|---:|
| Baseline aléatoire | `results/dev-pred-baseline.txt` | 4.43% (14/316) |
| Intersection char 3-gram, top 100 | `results/dev-pred-intersection-char-3gram-top100.txt` | 97.47% (308/316) |
| Naive Bayes simple (bag-of-words) | `results/dev-pred-naivebayes-bow.txt` | **99.68%** (315/316) |
| Naive Bayes n-grammes (char 3-gram, 2000) | `results/dev-pred-naivebayes-char-3gram-max2000.txt` | 98.73% (312/316) |
| kNN (modèle complet) | `results/dev-pred-knearest.txt` | 94.94% (300/316) |
| kNN (modèle rassemblé) | `results/dev-pred-knearest-gathered.txt` | 93.35% (295/316) |
| kNN avec TF-IDF | `results/dev-pred-knearest-idf-gathered.txt` | 94.30% (298/316) |
| Régression logistique (bag-of-words) | `results/dev-pred-logreg.txt` | 96.52% (305/316) |
| langid.py | `results/dev-pred-langid.txt` | 98.73% (312/316) |
| Compte des lettres | `results/dev-gathered-pred-knearest-letter.txt`| 18.99% (60/316) |


On observe que les modèles Naive Bayes (simple bag-of-words ou n-grammes de caractères) obtiennent les meilleures performances sur ce jeu de données, légèrement devant les autres classifieurs linéaires ou à base de k plus proches voisins. Les méthodes externes comme `langid.py` restent compétitives, mais sans avantage clair par rapport aux modèles entraînés spécifiquement sur ce corpus.

Nous avons notamment constaté des ralentissements lorsqu'on utilisait kNN tel qu'implémenté en TP. Ainsi nous avons fait le choix d'implémenter un modèle qui rassemble tous les textes d'une même langue sous un même vecteur pour chacun des textes du corpus d'entraînement. Au vu de la perte négligeable de résultats comparée au gain en terme de vitesse de génération du fichier, nous avons choisi de faire nos autres tests qui utilisent kNN avec ce modèle rassemblé. De plus, nous avons remarqué que TF-IDF n'apporte pas d'améliorations significatives ; on peut conjecturer que cela est notamment dû au fait que les langues sont très différentes entre elles pour la plupart, et donc que donner plus de poids aux mots les plus rares n'est pas un facteur d'amélioration significatif. On peut également noter que le compte de lettres donne des prédictions assez limitées par rapport aux autres modèles que nous avons développés, ce qui montre bien les apports donnés par les méthodes de TAL vis-à-vis de l'approche naïve. 

---

## Limitations

Le prototype présente quelques limites. Le corpus utilisé contient des phrases relativement bien formées ; les performances peuvent donc baisser sur des textes plus bruités (messages courts, réseaux sociaux, fautes fréquentes). Le système ne traite pas explicitement les cas de mélange de langues dans une même phrase. Enfin, la méthode kNN devient plus coûteuse en temps de calcul lorsque la taille du corpus augmente.

