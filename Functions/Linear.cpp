//
// Created by mathieu.hannoun on 09/11/2018.
//

#include "Linear.h"
#include "../Neurons/Neuron.h"

float Linear::usualFunction(std::vector<float> *inputs, Neuron *neuron) {
    return this->computeY(inputs, neuron);
}

float Linear::derivative(std::vector<float> *inputs, Neuron *neuron, float currentInput) {
    return currentInput;
}


