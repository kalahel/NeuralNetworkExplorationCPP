//
// Created by mathieu.hannoun on 09/11/2018.
//

#ifndef NEURALNETWORKEXPLORATION_NEURON_H
#define NEURALNETWORKEXPLORATION_NEURON_H
#define  LEARNING_RATE 0.01

#include <vector>  //for std::vector
#include <stdlib.h>

class ActivationFunction;

class Neuron {
private:
    std::vector<float> weights;
    ActivationFunction *activationFunction;

    /**
    * Initialize random weights
    * Bias is included so it allow weightsNumber - 1 inputs
    * @param weightsNumber
    */
    void initWeights(int weightsNumber);

public:
    Neuron(ActivationFunction *activationFunction, int weightsNumber);

    float compute(std::vector<float> *inputs);

    /**
    * Train neuron weights for one iteration
    * @param inputs
    * @param expectedOutput
    * @return Error with the new weights values
    */
    float trainWeights(std::vector<float> *inputs, float expectedOutput);

    /**
     * Train neuron weights for Multiple examples
     * NOT FUNCTIONNAL THE TRAINING FUNCTION IS WRONG
     * @param inputs
     * @param expectedOutput
     * @return Error with the new weights values
     */
    float trainWeightsMultipleExample(std::vector<std::vector<float>> *inputs, std::vector<float> *expectedOutput);

    const std::vector<float> &getWeights() const;

    void printWeights();

    virtual ~Neuron();
};


#endif //NEURALNETWORKEXPLORATION_NEURON_H
