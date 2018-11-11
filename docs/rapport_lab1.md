# Projet 1 : détection de visages avec des histogrammes de couleurs de peau

Le rapport présente tout d'abord l'implémentation faite pour les trois challenges, puis présente une analyse des performances afin de déterminer les paramètres optimaux correspondant aux trois challenges et comparer l'efficacité de nouvelles options.

## I. Les choix d'implémentation

L'algorithme de détection de visages par inférence Bayésienne se décompose en trois étapes i.e _challenges_.

Le premier _challenge_ attribue à chaque pixel $I(i,j)$ d'une image en couleur $I$ une probabilité $P(i,j)$ qu'il soit de la peau. Pour calculer cette probabilité, on utilise deux histogrammes : $H, H_T$ où $H$ compte l'ensemble des pixels dans le set d'images utilisés pour l'entraînement et $H_T$ compte les pixels de peau dans ce _data set_. Les performances du _challenge 1_ dépendent des paramètres suivants :

* $Q$ : facteur de quantification des histogrammes
* représentation des couleurs : espace couleur RGB et espace chrominance rg

Le _challenge 2_ désigne le passage d'une fenêtre coulissante afin de repérer dans l'image des probabilités $P(i,j)$ les Régions d'Intérêt (ROI) pouvant contenir un visage. Nous avons choisi d'utiliser directement une fenêtre elliptique afin de ne pas se soucier des valeurs situées aux bords de la fenêtre carrée. Nous avons paramétré plusieurs parcours de la fenêtre coulissante avec un nombre de tailles d'ellipses et un nombre d'orientation pour chaque taille. Les performances du _challenge 2_ dépendent des paramètres suivants :

* $(w, h)$ : largeur et hauteur de l'ellipse d'origine
* $\# scales$ : nombre d'échelles d'ellipses à partir de l'ellipse d'origine
* $\# angles$ : nombre d'angles (en degré) choisi entre 0° et 180°
* $B$ : valeur du biais

L'ensemble de visages détectés obtenu par le _challenge 2_ doit ensuite être traité afin de garder un maximum de visages différents. Pour cela, nous avons choisi de comparer deux approches différentes : la première est la suppression du non-maximum présentée dans le sujet, où l'on ne garde que les ellipses avec la plus grande vraissemblance en définissant une distance pour déterminer le voisinage entre deux ellipses. La deuxième consiste à utiliser une distance entre ellipses afin de les regrouper pour prendre l'enveloppe convexes de tous les groupes d'ellipses formés. Les performances du _challenge 3_ dépendent des paramètres suivants :

* choix de la distance entre ellipse : distance euclidienne vs. distance de Mahalanobis
* $R$ : distance maximale pour que deux ellipses soient considérées comme "voisines"

## II. Mesurer les performances

\# TODO : petite partie sur les métriques utilisées, et comment les calculer.

## III. Analyse des résultats

### 1. Paramètres du _challenge 1_

\# TODO : graphes des métriques avec la quantification $Q$ et le mode de couleurs => en déduire la meilleure quantification et le meilleur mode

### 2. Paramètres du _challenge 2_

\# TODO : même chose avec nombre d'angles et tailles initiales

### 3. Paramètres du _challenge 3_
\# TODO : même chose avec distance $R$ et comparer entre supprimer les ellipses et les regrouper en enveloppes convexes.
