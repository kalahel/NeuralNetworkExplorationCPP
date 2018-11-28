#include <iostream>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <random>
#include <time.h>
#include <chrono>
#include "Functions/Linear.h"
#include "Neurons/Neuron.h"

#define TRAINING_ITERATION 1000
using namespace std;

void simpleNeuralExample();

void trainNeuralMultiple();

std::vector<float> generateNoise(std::vector<float> *inputs, int numberOfModification);

void displayImage(std::vector<float> *inputs);

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

    //displayImage(&representationA);

    vector<float> noisyA = generateNoise(&representationC, 10);
    displayImage(&noisyA);


    for (int i = 0; i < TRAINING_ITERATION; ++i) {
        //std::cout << "Computation : " << neuronA.compute(&representationA) << std::endl;
        error = neuronA.trainWeights(&representationA, expectedValueA);
        error += neuronA.trainWeights(&representationC, expectedValueC);
        //std::cout << "New error " << error << std::endl;
        //errorFile << error << ",";
    }

    std::cout << "Computation : " << neuronA.compute(&noisyA) << std::endl;

    // Computation of average value for increasingly noisy images
    vector<float> noisyImage;
    for (int j = 1; j < 21; ++j) {
        float averageValue = 0.0;
        noisyImage = generateNoise(&representationC, j);
        for (int i = 0; i < 100; ++i) {
            averageValue += neuronA.compute(&noisyImage);
        }
        averageValue = averageValue / 100;
        errorFile << averageValue << ",";
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
    srand(static_cast<unsigned int>(getpid() * time(NULL)));
    vector<float> result = {};
    vector<int> indexToModify = {};
    for (int j = 0; j < numberOfModification; ++j) {
        indexToModify.push_back((int) (random() % inputs->size()));
    }
    for (int i = 0; i < inputs->size(); ++i) {
        result.push_back((*inputs)[i]);
    }
    for (int k = 0; k < indexToModify.size(); ++k) {
        if (result[indexToModify[k]] == 1.0)
            result[indexToModify[k]] = 0.0;
        else
            result[indexToModify[k]] = 1.0;
    }
    return result;
}

void displayImage(std::vector<float> *inputs) {
    for (int i = 0; i < inputs->size(); ++i) {
        std::cout << (*inputs)[i] << std::flush;
        if ((i + 1) % 4 == 0) {
            std::cout << std::endl;
        }
    }
}

