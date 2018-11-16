cmake_minimum_required(VERSION 3.12)
project(NeuralNetworkExploration)

set(CMAKE_CXX_STANDARD 14)

add_executable(NeuralNetworkExploration main.cpp Functions/ActivationFunction.cpp Functions/ActivationFunction.h Functions/Linear.cpp Functions/Linear.h Neurons/Neuron.cpp Neurons/Neuron.h)