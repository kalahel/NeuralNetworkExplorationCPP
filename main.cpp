#include <iostream>
#include "Functions/Linear.h"
#include "Neurons/Neuron.h"

using namespace std;

void simpleNeuralExample();
void trainNeuralMultiple();

int main() {



}

void simpleNeuralExample() {
    Linear linear;
    Neuron neuron(&linear, 3);
    neuron.printWeights();
    vector<float> inputs = {0.0, 1.0};

    float expectedValue = 1.0;
    float error = 10.0;
    int iteration_number = 0;
    while (error > 0.0001) {
        std::cout << "Computation : " << neuron.compute(&inputs) << std::endl;
        error = neuron.trainWeights(&inputs, expectedValue);
        std::cout << "New error " << error << std::endl;
        neuron.printWeights();
        iteration_number++;
    }
    std::cout << "Number of iterations until convergence : " << iteration_number << std::endl;

}

// NOT WORKING
void trainNeuralMultiple(){
    Linear linear;
    Neuron neuron(&linear, 3);
    neuron.printWeights();

    std::vector<std::vector<float>> inputs;
    inputs.push_back({0.0,0.0});
    inputs.push_back({0.0,1.0});
    inputs.push_back({1.0,0.0});
    inputs.push_back({1.0,1.0});

    vector<float> expectedValue = {0.0,0.0,0.0,1.0};
    // NOT FUNCTIONAL
    float error = 10.0;
    int iteration_number = 0;
    while (error*error > 0.0001) {
        error = neuron.trainWeightsMultipleExample(&inputs, &expectedValue);
        std::cout << "New error " << error << "\n" << std::endl;
        neuron.printWeights();
        iteration_number++;
    }
    std::cout << "Number of iterations until convergence : " << iteration_number << std::endl;

}