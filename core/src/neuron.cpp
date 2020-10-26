#include "neuron.hpp"

namespace SNN
{ 
    Neuron::Neuron(std::string name, double a, double b, double c, double d, uint32_t index) {
        this->name = name;
        this->a = a;
        this->b = b;
        this->c = c;
        this->d = d;
        this->index = index;

        this->prevV = -70;
        this->prevU = -14;
    } 

    double Neuron::AddCurrent(double i)
    {
        this->current += i;
        return this->current;
    }


    std::pair<double, double> Neuron::CalculatePotential() {
        double Vans, Uans;
        if(this->prevV < 35) {
            double dv = (0.04 * this->prevV + 5) * this->prevV + 140 - this->prevU;
            Vans = this->prevV + (dv + this->current) * this->dt;
            double du = this->a * (this->b * this->prevV - this->prevU);
            Uans = this->prevU + this->dt * du;
            if(Vans > 35)
                Vans = 35;
        }
        else {
            Vans = this->c;
            Uans = this->prevU + this->d;
        }
        this->prevU = Uans;
        this->prevV = Vans;
        this->current = 0;
        return std::pair(Vans, Uans);
    }
}