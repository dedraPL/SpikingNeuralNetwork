#include "synapse.hpp"

namespace SNN
{ 
    Synapse::Synapse(Node* src, Node* dest, SYNAPSE_TYPE r) {
        this->r = r;
        this->dest = dest;
        this->src = src;
    }

    SYNAPSE_TYPE Synapse::CalculateCurrent(SYNAPSE_TYPE v1, SYNAPSE_TYPE v2) {
        if(v2 < v1) {
            return ((v1 - v2) / this->r);
        }
        return 0;
    }

    void Synapse::ChangeResistance(SYNAPSE_TYPE res) {
        this->r = res;
    }
}