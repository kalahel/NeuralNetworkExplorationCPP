//
// Created by mathieu.hannoun on 09/11/2018.
//

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
                    (error) * this->activationFunction->derivative(inputs, this, 1) * LEARNING_RATE;    // Biais
        } else {
            this->weights[i] +=
                    error * this->activationFunction->derivative(inputs, this, (*inputs)[i - 1]) * LEARNING_RATE;
        }
    }
    return error;
}

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



