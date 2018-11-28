# TD IA 3

## Questions préliminaires

Les lettres seront représenté par des tableaux 4*5 soit :

```
	####
	#   
c = #
	#
	####
```

```
	#### 
	#  #
a =	####
	#  #
	#  #
```

Les tableaux seront passés en une dimension pour question de simplicité :

```c++
c = {1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1}
a = {1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1}
```

### Représentation du réseau

![NeuralNetworkRepresentation](.\assets\NeuralNetworkRepresentation.png)

Ceci est une représentation du réseau de neurone que nous utiliserions si nous voulions classifier indépendamment chaque lettre avec  un neurone et une fonction de Heaviside.

Voici le réseau de neurone demandé :



![Réseau de Neurone](http://image.noelshack.com/fichiers/2018/48/3/1543408858-oneneuronrepresentation.png)



### Implémentation d'un neurone

Nous allons programmer en C++ ce qui nous permet d'avoir une approche orientée objet.

Chaque neurone intègrera une fonction d'activation (on utilisera le polymorphisme du C++ pour pouvoir appliquer n'importe quelle fonction d'activation)

Pour chaque neurone on aura :

$y = \sum{w_i * e_i}$

$S = f(y)$ avec $f$ la fonction d'activation.

Ici nous pourrions utiliser une fonction de Heaviside tel que :

$\forall y\in \mathbb{R} ,\ H(y)=\left\{{\begin{matrix}0&{\mathrm  {si}}&y<0\\1&{\mathrm  {si}}&\geq 0.\end{matrix}}\right.$

Mais nous avons préféré utiliser une fonction purement linéaire pour ce TP :

$f(y)= y$

### Initialisation des poids

On initialisera les poids de façon aléatoire avec des valeurs flottantes comprises entre 0 et 1.

### Propagation de l'information

L'information se propagera des entrées vers la première couche de neurones puis de couches de couches (la sortie de la couche *i* est l'entré de la couche i+1) jusqu'à la dernière couche pour produire une sortie.

### Variation des poids

les poids d'un neurone serons ajusté en fonction des sorties attendues comme tel :

$W_{i (t+1)} = W_{i(t)} + \epsilon * (d - S) * e_i$

- $\epsilon$ : constante apprentissage
- $d$ : valeur attendue
- $S$ : sortie du neurone
- $e_i$ : entrée $i$ du neurone   

### Durée apprentissage

La vitesse d'apprentissage est entièrement dépendante de la valeur de $\epsilon$ généralement fixé à $0.1$.

Plus $\epsilon$ est élevé plus l'apprentissage sera rapide mais il gagnera aussi en instabilité.

## Questions de compréhension	

### 1) Qu'effectue l'apprentissage ?

L'apprentissage en regardant la matrice de poids définira le liens entre les entrées et la sortie d'un neurone, modifiant les poids pour faire correspondre le résultat obtenu à celui attendu.

