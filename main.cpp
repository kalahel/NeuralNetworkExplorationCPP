#include <iostream>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include "Functions/Linear.h"
#include "Neurons/Neuron.h"

#define TRAINING_ITERATION 1000
using namespace std;

void simpleNeuralExample();

void trainNeuralMultiple();

std::vector<float> generateNoise(std::vector<float> *inputs,int numberOfModification);

int main() {

    ofstream errorFile;
    errorFile.open("../assets/errorEvolution.csv");
    if (!errorFile.is_open()) {
        char cwd[256];
        cerr << "errorFile isn't open !" << endl;
        cerr << strerror(errno) << endl;
        cerr << "We work in : " << getcwd(cwd, sizeof(cwd)) << endl;
    }
    vector<float> representationA = {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1};
    vector<float> representationC = {1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};

    Linear linear;
    Neuron neuronA(&linear, (int) representationA.size() + 1);
    float expectedValueA = 1.0;
    float expectedValueC = 0.0;

    float error = 10.0;

    for (int i = 0; i < TRAINING_ITERATION; ++i) {
        //std::cout << "Computation : " << neuronA.compute(&representationA) << std::endl;
        error = neuronA.trainWeights(&representationA, expectedValueA);
        error += neuronA.trainWeights(&representationC, expectedValueC);
        std::cout << "New error " << error << std::endl;
        errorFile << error << ",";
    }

    errorFile.close();

}

void trainUntilConvergeance() {
    vector<float> representationA = {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1};


    Linear linear;
    Neuron neuronA(&linear, (int) representationA.size() + 1);
    neuronA.printWeights();

    float expectedValue = 1.0;
    float error = 10.0;
    int iteration_number = 0;
    while (!(error < 0.0001 && error > -0.0001)) {
        std::cout << "Computation : " << neuronA.compute(&representationA) << std::endl;
        error = neuronA.trainWeights(&representationA, expectedValue);
        std::cout << "New error " << error << std::endl;
        neuronA.printWeights();
        iteration_number++;
    }
    std::cout << "Number of iterations until convergence : " << iteration_number << std::endl;

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
void trainNeuralMultiple() {
    Linear linear;
    Neuron neuron(&linear, 3);
    neuron.printWeights();

    std::vector<std::vector<float>> inputs;
    inputs.push_back({0.0, 0.0});
    inputs.push_back({0.0, 1.0});
    inputs.push_back({1.0, 0.0});
    inputs.push_back({1.0, 1.0});

    vector<float> expectedValue = {0.0, 0.0, 0.0, 1.0};
    // NOT FUNCTIONAL
    float error = 10.0;
    int iteration_number = 0;
    while (error * error > 0.0001) {
        error = neuron.trainWeightsMultipleExample(&inputs, &expectedValue);
        std::cout << "New error " << error << "\n" << std::endl;
        neuron.printWeights();
        iteration_number++;
    }
    std::cout << "Number of iterations until convergence : " << iteration_number << std::endl;

}

std::vector<float> generateNoise(std::vector<float> *inputs, int numberOfModification) {
    vector<float> result = {};
    for (int i = 0; i < inputs->size(); ++i) {
        
    }
    return result;
}

