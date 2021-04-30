#include "neuron.hpp"

namespace SNN { 
    Neuron::Neuron(NEURON_TYPE a, NEURON_TYPE b, NEURON_TYPE c, NEURON_TYPE d)
    {
        this->a = a;
        this->b = b;
        this->c = c;
        this->d = d;

        this->prevV = -70;
        this->prevU = -14;
    } 

    NEURON_TYPE Neuron::AddCurrent(NEURON_TYPE i)
    {
        this->current += i;
        return this->current;
    }

    std::pair<NEURON_TYPE, NEURON_TYPE> Neuron::CalculatePotential()
    {
        NEURON_TYPE Vans, Uans;
        if(this->prevV < 35) 
        {
            NEURON_TYPE dv = (0.04 * this->prevV + 5) * this->prevV + 140 - this->prevU;
            Vans = this->prevV + (dv + this->current) * this->dt;
            NEURON_TYPE du = this->a * (this->b * this->prevV - this->prevU);
            Uans = this->prevU + this->dt * du;
            if(Vans > 35)
                Vans = 35;
        }
        else 
        {
            Vans = this->c;
            Uans = this->prevU + this->d;
        }
        this->prevU = Uans;
        this->prevV = Vans;
        this->current = 0;
        return std::pair(Vans, Uans);
    }
}