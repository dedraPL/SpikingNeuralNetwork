#pragma once

#define NEURON_TYPE float

#include <string>

namespace SNN {
    class Neuron 
    {
    public:
        std::string name;
        NEURON_TYPE a, b, c, d;
        uint32_t index;
        NEURON_TYPE prevV, prevU;
        NEURON_TYPE current = 0;

        Neuron(std::string name, NEURON_TYPE a, NEURON_TYPE b, NEURON_TYPE c, NEURON_TYPE d, uint32_t index = 0);
        ~Neuron() {};

        NEURON_TYPE AddCurrent(NEURON_TYPE i);
        std::pair<NEURON_TYPE, NEURON_TYPE> CalculatePotential();
    private:
        NEURON_TYPE dt = 0.5;
    };

}