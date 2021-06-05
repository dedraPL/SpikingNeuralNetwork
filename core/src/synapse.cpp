#include "synapse.hpp"

namespace SNN
{ 
    Synapse::Synapse(Node* src, Node* dest, SYNAPSE_TYPE r) {
        this->r = r;
        this->dest = dest;
        this->src = src;
        this->C = new SYNAPSE_TYPE[1];
        this->prevV = 0;
    }

    SYNAPSE_TYPE Synapse::CalculateCurrent(SYNAPSE_TYPE v1, SYNAPSE_TYPE v2) {
        if(v2 < v1) {
            SYNAPSE_TYPE v = ((v1 - v2) / this->r) + ((v1 - v2) - prevV) * this->C[0];
            this->prevV = (v1 - v2);
            return v;
        }
        return 0;
    }

    void Synapse::ChangeResistance(SYNAPSE_TYPE res) {
        this->r = res;
    }

    void Synapse::ChangeCapacitance(SYNAPSE_TYPE c) {
        this->C[0] = c;
    }
}