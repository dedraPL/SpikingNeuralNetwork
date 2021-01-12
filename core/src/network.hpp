#pragma once

#include <string>
#include <map>
#include <vector>
#include <list>
#include <exception>
#include <algorithm>

#include "neuron.hpp"
#include "synapse.hpp"

namespace SNN {
    class Network {
    public:
        enum class NodeMode { input, output, hidden };
        
        struct Node {
            Neuron* node;
            NodeMode mode;
            std::vector<Synapse*> conn;
            std::vector<uint32_t> sources;

            void update(Neuron* node, NodeMode mode, std::vector<Synapse*> conn) {
                this->node = node;
                this->mode = mode;
                this->conn = conn;
            }

            Neuron getNode() { return *node; }
            NodeMode getMode() { return mode; }
            std::vector<Synapse*> getConn() { return conn; }
            std::vector<uint32_t> getSources() { return sources; }
        };

        class InputSizeError : public std::exception {
            std::string _msg;
        public:
            explicit InputSizeError(const int& inputSize)
            {
                _msg = std::string("input size is not valid, expected ") + std::to_string(inputSize);
            }
            const char* what() const throw () {
                return _msg.c_str();
            }
        };

        Network();
        std::vector<std::vector<uint32_t>>* BFSSort();
        std::vector<double> run(std::vector<double> inputs);

        std::map<uint32_t, Network::Node*> graph;
        std::map<uint32_t, Network::Node*> getGraph() { return graph; };
        uint32_t inputSize = 0, outputSize = 0;
        std::vector<std::vector<uint32_t>> graphOrder;
    private:
    };
}