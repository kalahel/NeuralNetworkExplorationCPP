//
// Created by mathieu.hannoun on 09/11/2018.
//

#ifndef NEURALNETWORKEXPLORATION_ACTIVATIONFUNCTION_H
#define NEURALNETWORKEXPLORATION_ACTIVATIONFUNCTION_H

#include <vector>  //for std::vector
#include <stdio.h>
#include <stdlib.h>

class Neuron;

class ActivationFunction {
public:
    virtual float usualFunction(std::vector<float> *inputs, Neuron *neuron) = 0;

    virtual float derivative(std::vector<float> *inputs, Neuron *neuron, float currentInput) = 0;

    float computeY(std::vector<float> *inputs, Neuron *neuron);

};


#endif //NEURALNETWORKEXPLORATION_ACTIVATIONFUNCTION_H
