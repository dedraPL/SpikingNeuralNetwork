#pragma once

#include <string>

namespace SNN {
    class Neuron {
    public:
        std::string name;
        double a, b, c, d;
        uint32_t index;
        double prevV, prevU;
        double current = 0;

        Neuron(std::string name, double a, double b, double c, double d, uint32_t index = 0);
        ~Neuron() {};

        double AddCurrent(double i);
        std::pair<double, double> CalculatePotential();
    private:
        double dt = 0.5;
    };

}