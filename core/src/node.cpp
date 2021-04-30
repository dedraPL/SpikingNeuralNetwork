#include "node.hpp"
namespace SNN {
    Node::Node(uint32_t name, uint32_t index) : mode(NodeMode::hidden), name(name), index(index) {
        node = std::make_shared<Neuron>(0, 0, 0, 0);
    }
}
