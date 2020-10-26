#include "synapse.hpp"

namespace SNN
{ 
    Synapse::Synapse(uint32_t dest, double r) {
        this->r = r;
        this->dest = dest;
    }

    double Synapse::CalculateCurrent(double v1, double v2) {
        if(v2 < v1) {
            return ((v1 - v2) / this->r);
        }
        return 0;
    }

    void Synapse::ChangeResistance(double res) {
        this->r = res;
    }
}