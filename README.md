Deviner la langue d'un texte
----------------------------

Le problème posé ici est : comment peut-on deviner automatiquement dans quelle langue ou dialecte un texte a été écrit ?

Un système d'identification automatique de la langue est utile, par exemple, dans un système de traduction automatique, pour savoir depuis quelle langue traduire. Il peut être utilisé également pour adapter les paramètres d'un système de reconnaissance vocale selon la langue de l'utilisateur. Les robots qui balaient le web utilisent des systèmes de détection de la langue pour catégoriser et indexer correctement les pages. On peut aussi utiliser un tel système pour détecter et ignorer des passages en langue étrangère lorsqu'on est en train d'analyser un texte en français.

L'identification de la langue est une tâche assez facile en TAL, comparée à la traduction automatique ou à la reconnaissance de la parole. Cependant, certains défis peuvent se poser pour distinguer les langues proches (p. ex. portugais et espagnol) et pour traiter des textes très courts comme les tweets, par exemple, qui ne dépassent pas les 140 caractères. 

## Catégories à prédire

Vous devez prédire une catégorie parmi les codes ISO ci-dessous des langues inclues dans le dataset :
  * bg - Bulgare
  * cs - Tchèque
  * da - Dannois
  * de - Allemand
  * el - Grec
  * en - Anglais
  * es - Espagnol
  * et - Estonian
  * fi - Finnois
  * fr - Français
  * hu - Hongrois
  * it - Italien
  * lt - Lituanian
  * lv - Letonian
  * nl - Néerlandais
  * pl - Polonais
  * pt - Portugais
  * ro - Roumain
  * sk - Slovaque
  * sl - Slovénien
  * sv - Suédois

Ce corpus a été créé spécialement pour ce projet à partir du corpus parallèle [Europarl](https://www.statmt.org/europarl/).

## Développement à faire

Vous devez écrire un logiciel qui devine, pour un texte donné en entrée, quelle est sa langue. Par exemple, si on donne en entrée le texte _Hallo, wie geht es dir?_ votre programme doit donner comme résultat `de`, qui est le code de la langue allemande. Cependant, si le texte en entrée est _Hola buenos días niño, ¿qué tal?_, il faut prédire `es` pour espagnol.

Votre système doit donner en sortie un fichier au même format que l'entrée, avec une copie du tweet suivie d'une tabulation suivie de la catégorie prédite.

## Extensions ou alternatives

Vous devez proposer des améliorations et/ou des extensions si vous travaillez uniquement sur ces données. Nous proposons soit de tester votre système sur un des autres datasets proposés, soit de faire une étude empirique qui mesure (a) la performance du système et (b) le temps d'exécution en faisant varier la taille du jeu de données d'entraînement et la longueur des textes à prédire. Pour cela, vous pouvez réduire artificiellement la taille des datasets et la longuer des phrases en supprimant tout ce qui dépasse un certain seuil. Vous créerez des datasets de taille/longueur incrémentale, et vous ferez une courbe pour montrer comment la performance et le temps d'exécution évoluent en fonction de la taille/longueur du dataset/des phrases.
