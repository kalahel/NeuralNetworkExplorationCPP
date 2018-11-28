# TD IA 3

Mathieu HANNOUN avec Matteo STAIANO

## Questions préliminaires

Les lettres seront représentées par des tableaux 4*5 soit :

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

Voici le réseau de neurone demandé et utilisé :



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

### Neurone

Classe d'un neurone

```c++
#include <iostream>
#include "Neuron.h"
#include "../Functions/ActivationFunction.h"

void Neuron::initWeights(int weightsNumber) {
    this->weights.clear();
    for (int i = 0; i < weightsNumber; i++) {
        int randInt = rand() % 100;         // In the range 0 to 99
        float randFloat = ((float) randInt) / 100;
        this->weights.push_back(randFloat);
    }
}

const std::vector<float> &Neuron::getWeights() const {
    return weights;
}

float Neuron::compute(std::vector<float> *inputs) {
    return this->activationFunction->computeY(inputs, this);
}

Neuron::Neuron(ActivationFunction *activationFunction, int weightsNumber) : activationFunction(activationFunction) {
    this->initWeights(weightsNumber);
}

void Neuron::printWeights() {
    for (int i = 0; i < this->weights.size(); ++i) {
        std::cout << "W" << i << " : " << weights[i] << std::endl;
    }
}

float Neuron::trainWeights(std::vector<float> *inputs, float expectedOutput) {
    float error = (expectedOutput - this->activationFunction->usualFunction(inputs, this));

    for (int i = 0; i < this->weights.size(); ++i) {
        if (i == 0) {
            this->weights[0] +=
                    (error) * this->activationFunction->derivative(inputs, this, 1) * LEARNING_RATE;    // Bias
        } else {
            this->weights[i] +=
                    error * this->activationFunction->derivative(inputs, this, (*inputs)[i - 1]) * LEARNING_RATE;
        }
    }
    return error;
}

// NOT FUNCTIONNAL
float Neuron::trainWeightsMultipleExample(std::vector<std::vector<float>> *inputs, std::vector<float> *expectedOutput) {
    if ((*inputs).size() != (*expectedOutput).size()) {
        perror("Inputs and expectedOutputs size missmatch");
        exit(1);
    }
    float errorSum = 0;
    for (int i = 0; i < (*inputs).size(); ++i) {
        errorSum += (*expectedOutput)[i] -
                    this->activationFunction->usualFunction((&(*inputs)[i]), this); // TODO CHECK IF CORRECT
    }

    for (int i = 0; i < this->weights.size(); ++i) {
        float derivativeSum = 0.0;
        if (i == 0) {
            for (int j = 0; j < (*inputs).size(); ++j) {
                derivativeSum += this->activationFunction->derivative((&(*inputs)[j]), this, 1);
            }
            this->weights[0] +=
                    (errorSum) * derivativeSum * LEARNING_RATE;    // Bias
        } else {
            for (int k = 0; k < (*inputs).size(); ++k) {
                derivativeSum += this->activationFunction->derivative((&(*inputs)[k]), this, (*inputs)[k][i - 1]);
            }
            this->weights[i] +=
                    errorSum * derivativeSum * LEARNING_RATE;
        }
    }

    return errorSum;
}

Neuron::~Neuron() = default;
```

### Fonction d'activation

Fonction d'activation abstraite permettant d'utiliser des concepts de **polymorphisme**.

```c++
#include "ActivationFunction.h"
#include "../Neurons/Neuron.h"

float ActivationFunction::computeY(std::vector<float> *inputs, Neuron *neuron) {
    if ((*inputs).size() + 1 != (*neuron).getWeights().size()) {
        perror("Sizes missmatch inputs and weights");
        exit(1);
    }
    float sum = (*neuron).getWeights()[0];
    for (int i = 0; i < (*inputs).size(); ++i) {
        sum += (*inputs)[i] * (*neuron).getWeights()[i + 1];
    }
    return sum;
}
```

#### Fonction linéaire

```c++
#include "ActivationFunction.h"
#include "../Neurons/Neuron.h"

float ActivationFunction::computeY(std::vector<float> *inputs, Neuron *neuron) {
    if ((*inputs).size() + 1 != (*neuron).getWeights().size()) {
        perror("Sizes missmatch inputs and weights");
        exit(1);
    }
    float sum = (*neuron).getWeights()[0];
    for (int i = 0; i < (*inputs).size(); ++i) {
        sum += (*inputs)[i] * (*neuron).getWeights()[i + 1];
    }
    return sum;
}
```

### Entrainement du réseau de neurone

```c++
vector<float> representationA = {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1};
vector<float> representationC = {1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};

Linear linear;
Neuron neuronA(&linear, (int) representationA.size() + 1);
float expectedValueA = 1.0;
float expectedValueC = 0.0;

float error = 10.0;

for (int i = 0; i < TRAINING_ITERATION; ++i) {
    std::cout << "Computation : " << neuronA.compute(&representationA) << std::endl;
    error = neuronA.trainWeights(&representationA, expectedValueA);
    error += neuronA.trainWeights(&representationC, expectedValueC);
    std::cout << "New error " << error << std::endl;
}
```

On entraine un nombre de fois donnée notre neurone tout en affichant la somme des erreurs obtenue à chaque itérations

#### Résultat

Pour 100 itérations d'apprentissage :

![Apprentissage](http://image.noelshack.com/fichiers/2018/48/3/1543415353-chart.jpg)



Après 1000 itérations on obtiens une erreur de $-7.45058e^{-8}$.

Généralement le réseau atteint une erreur de $0.0001$ après une soixantaine d'itérations.

### Bruitage de l'image

Nous allons tester notre neurone avec des images bruités, voici notre fonction de bruitage :

```c++
std::vector<float> generateNoise(std::vector<float> *inputs, int numberOfModification) {
    srand(static_cast<unsigned int>(getpid() * time(NULL)));
    vector<float> result = {};
    vector<int> indexToModify = {};
    for (int j = 0; j < numberOfModification; ++j) {
        indexToModify.push_back((int)(random() % inputs->size()));
    }
    for (int i = 0; i < inputs->size(); ++i) {
       result.push_back((*inputs)[i]);
    }
    for (int k = 0; k < indexToModify.size(); ++k) {
        if(result[indexToModify[k]] == 1.0)
            result[indexToModify[k]] = 0.0;
        else
            result[indexToModify[k]] = 1.0;
    }
    return result;
}
```

Voici les résultats obtenues :

#### Pour 1 pixel modifié :

**A**

```
1110
1001
1111
1001
1001

Computation : 1.33262
```

**C**

```
1110
1000
1000
1000
1111

Computation : 0.332625
```

#### Pour 2 pixels modifiés :

**A**

```
1110
1011
1111
1001
1001

Computation : 1.89262
```

**C**

```
1110
1010
1000
1000
1111

Computation : 0.892625
```

#### Pour 3 pixels modifiés :

**A**

```
1110
1011
1111
1001
1101

Computation : 2.33075
```

**C**

```
1110
1010
1000
1000
1011

Computation : 0.454499
```

#### Pour 10 pixels modifiés :

**A**

```
1010
1001
1011
0101
1101

Computation : 1.81675
```

**C**

```
1010
1000
1100
0100
1011

Computation : -0.001
```



Evidement plus le bruit est élevé plus les résultats tendent à diverger des valeurs 0 & 1.

En utilisant une fonction de Heaviside nous pourrions avoir une caractérisation évidente des lettres jusqu'à un certain point, car les seuls résultats possible seraient la détection de la lettre C ou A sans indicateur clair de précision.

#### Evolution des sorties en fonctions du bruits

Nous avons faits des tests pour montrer l'évolutions des sorties du neurone essayant de reconnaitre une lettre bruitée.

Sur les abscisses le nombres de "bruit" ajouté à la lettre (le nombre de 1 changé en 0 et inversement).

Pour chaque valeur de bruit ajouté on a calculé la sortie $100$ fois et moyennés ces résultats.

##### Fonction réalisant ces calculs

```c++
vector<float> noisyImage;
for (int j = 1; j < 21; ++j) {
    float averageValue = 0.0;
    noisyImage = generateNoise(&representationC, j);	// Ici c'est C qui est bruité
    for (int i = 0; i < 100; ++i) {
        averageValue += neuronA.compute(&noisyImage);
    }
    averageValue = averageValue / 100;
    errorFile << averageValue << ",";
}
```

##### Pour A

![A bruitée](http://image.noelshack.com/fichiers/2018/48/3/1543416521-abruit.png)

#### Pour C

![C bruitée](http://image.noelshack.com/fichiers/2018/48/3/1543416606-cbruit.png)

Les résultats, malgré le fait de les moyenner,  restent assez complexe à expliqué.

Si au départ les résultats sont corrects (0 et 1) il y a ensuite inversion des deux lettre puis une instabilité croissante.

Ces résultats peuvent être expliqués par l'aléatoire introduite dans notre façon de bruiter.

## Questions de compréhension	

### 1) Qu'effectue l'apprentissage ?

L'apprentissage en regardant la matrice de poids définira le liens entre les entrées et la sortie d'un neurone, modifiant les poids pour faire correspondre le résultat obtenu à celui attendu.

### 2) Robustesse du réseau en cas de translation/rotation des entrées

Le réseau ayant appris "par cœur" à identifier les différentes lettres est extrêmement sensible à une rotation/translation des entrés, en effet le réseau reconnait les lettre en fonction des pixel activés ou non, toute modification de la position de la lettre rendrait le réseau incapable de la reconnaitre.

### 3) Reconnaissance de tout l'alphabet

Une méthode naïve mais pourtant efficace serait utiliser un réseau d'une seul couche composé de 26 neurones, chaque neurone serait responsable d'identifier une seule lettre.