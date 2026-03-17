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
Chaque texte est représenté sous forme de vecteur de fréquences, puis comparé aux exemples d’entraînement avec une similarité cosinus. La classe est décidée à partir des voisins les plus proches. Cette méthode est une bonne référence classique, mais elle est plus coûteuse en prédiction.

4. **Naive Bayes (scikit-learn)**  
Les textes sont vectorisés avec `CountVectorizer` (n-grammes mots ou caractères), puis un modèle `MultinomialNB` est entraîné. En pratique, c’est un très bon compromis entre simplicité, vitesse et performance sur ce dataset, même si l’hypothèse d’indépendance entre features reste approximative.

5. **Comparaison externe (`langid`)**  
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
│   └── naive_bayes.py
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

### Naive Bayes (char 3-gram, 2000 features)

```bash
python classifiers/naive_bayes.py train.txt dev.txt char 3 2000
python eval.py results/dev-pred-naivebayes-char-3gram-max2000.txt dev.txt
```

### kNN

```bash
python classifiers/knn_train.py train.txt
python classifiers/knn_predict.py dev.txt
python eval.py results/dev-pred-knearest-gathered.txt dev.txt
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

Exemple avec Naive Bayes :

```bash
python classifiers/naive_bayes.py train.txt test.txt char 3 2000
```

Fichier généré :

- `results/test-pred-naivebayes-char-3gram-max2000.txt`

Ce fichier contient la même structure que `test.txt`, avec `??` remplacé par les labels prédits.

---

## Résultats obtenus

Mesures observées sur `dev.txt` :

| Fichier de prédiction | Accuracy |
|---|---:|
| `results/dev-pred-bayes.txt` | **99.68%** (315/316) |
| `results/dev-pred-knearest.txt` | 94.94% (300/316) |
| `results/dev-pred-knearest-gathered.txt` | 93.35% (295/316) |
| `results/dev-pred1.txt` | 68.67% (217/316) |

On observe que le modèle Naive Bayes avec des n-grammes de caractères obtient la meilleure performance. Ce résultat est cohérent avec la tâche : les n-grammes capturent des motifs orthographiques propres à chaque langue, ce qui facilite la discrimination entre classes.

---

## Limitations

Le prototype présente quelques limites. Le corpus utilisé contient des phrases relativement bien formées ; les performances peuvent donc baisser sur des textes plus bruités (messages courts, réseaux sociaux, fautes fréquentes). Le système ne traite pas explicitement les cas de mélange de langues dans une même phrase. Enfin, la méthode kNN devient plus coûteuse en temps de calcul lorsque la taille du corpus augmente.

