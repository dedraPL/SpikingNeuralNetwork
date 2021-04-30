#pragma once

#define SYNAPSE_TYPE float

#include "node.hpp"

namespace SNN {
    class Node;

    class Synapse {
    public:
        SYNAPSE_TYPE r;
        Node* src;
        Node* dest;
        
        Synapse(Node* src, Node* dest, SYNAPSE_TYPE r);

        SYNAPSE_TYPE CalculateCurrent(SYNAPSE_TYPE v1, SYNAPSE_TYPE v2);
        void ChangeResistance(SYNAPSE_TYPE res);
    };

}