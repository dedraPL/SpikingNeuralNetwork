#pragma once

#include <string>

namespace SNN {
    class Synapse {
    public:
        double r;
        uint32_t dest;
        
        Synapse(uint32_t dest, double r);

        double CalculateCurrent(double v1, double v2);
        void ChangeResistance(double res);
    };

}