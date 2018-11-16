//
// Created by mathieu.hannoun on 09/11/2018.
//

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
