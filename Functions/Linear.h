//
// Created by mathieu.hannoun on 09/11/2018.
//

#ifndef NEURALNETWORKEXPLORATION_LINEAR_H
#define NEURALNETWORKEXPLORATION_LINEAR_H


#include "ActivationFunction.h"

class Linear : public ActivationFunction{
public:
    float usualFunction(std::vector<float> *inputs, Neuron *neuron) override;

    float derivative(std::vector<float> *inputs, Neuron *neuron, float currentInput) override;
};


#endif //NEURALNETWORKEXPLORATION_LINEAR_H
