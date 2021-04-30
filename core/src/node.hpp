#pragma once

#include <vector>
#include <memory>

#include "neuron.hpp"
#include "synapse.hpp"

namespace SNN {
    class Synapse;

    class Node {
    public:
        enum class NodeMode { input, output, hidden };

        std::shared_ptr<Neuron> node;
        NodeMode mode;
        std::vector<std::shared_ptr<Synapse>> conn;
        std::vector<std::shared_ptr<Synapse>> sources;
        uint32_t name;
        uint32_t index;

        Node(uint32_t name, uint32_t index = 0);

        std::shared_ptr<Neuron> getNode() { return node; }
        NodeMode getMode() { return mode; }
        std::vector<std::shared_ptr<Synapse>> getConn() { return conn; }
        std::vector<std::shared_ptr<Synapse>> getSources() { return sources; }
    };
}