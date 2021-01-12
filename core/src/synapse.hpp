#pragma once

#define SYNAPSE_TYPE float

#include <string>

namespace SNN {
    class Synapse {
    public:
        SYNAPSE_TYPE r;
        uint32_t dest;
        
        Synapse(uint32_t dest, SYNAPSE_TYPE r);

        SYNAPSE_TYPE CalculateCurrent(SYNAPSE_TYPE v1, SYNAPSE_TYPE v2);
        void ChangeResistance(SYNAPSE_TYPE res);
    };

}