#include <vector>

#include "NetworkEditor.hpp"

namespace SNN {
    std::shared_ptr<Neuron> NetworkEditor::addHiddenNode(Network& network)
    {
        return addNode(network, 0, Network::NodeMode::hidden);
    }

    std::shared_ptr<Neuron> NetworkEditor::addNode(Network& network, uint32_t index, Network::NodeMode mode)
    {
        uint32_t name = network.graph.size();
        Neuron* node = new Neuron(std::to_string(name), 0.02, 0.2, -65, 8, index);

        network.graph.insert({ name, std::make_shared<Network::Node>() });
        std::vector<std::shared_ptr<Synapse>> conn;
        network.graph[name]->update(*node, mode, conn);

        if (mode == Network::NodeMode::input) 
        {
            network.inputSize++;
        }
        else if (mode == Network::NodeMode::output)
        {
            network.outputSize++;
        }

        return network.graph[name]->node;
    }

    std::shared_ptr<Synapse> NetworkEditor::addSynapse(Network& network, Neuron& source, Neuron& destination, SYNAPSE_TYPE r)
    {
        uint32_t dest = std::stoi(destination.name);
        uint32_t src = std::stoi(source.name);
        std::shared_ptr<Synapse> synapse = std::make_shared<Synapse>(dest, r);
        network.graph[src]->conn.push_back(synapse);
        network.graph[dest]->sources.push_back(src);
        return synapse;
    }

    void NetworkEditor::removeNode(Network& network, Network::Node& node)
    {
        {
            auto input = std::begin(node.sources);
            uint32_t dest = std::stoi(node.node->name);
            while (input != std::end(node.sources))
            {
                uint32_t src = *input;

                for (auto it = std::begin(network.graph[src]->conn); it != std::end(network.graph[src]->conn); ++it)
                {
                    if ((*it)->dest == dest)
                    {
                        network.graph[src]->conn.erase(it);
                        break;
                    }
                }
                input = node.sources.erase(input);
            }
        }

        {
            auto output = std::begin(node.conn);
            uint32_t src = std::stoi(node.node->name);
            while (output != std::end(node.conn))
            {
                uint32_t dest = std::stoi(network.graph[(*output)->dest]->node->name);

                output = node.conn.erase(output);

                for (auto it = std::begin(network.graph[dest]->sources); it != std::end(network.graph[dest]->sources); ++it)
                {
                    if ((*it) == src)
                    {
                        network.graph[dest]->sources.erase(it);
                        break;
                    }
                }
            }
        }

        for (auto it = std::begin(network.graph); it != std::end(network.graph); ++it)
        {
            if (it->second.get() == &node)
            {
                network.graph.erase(it);
                break;
            }
        }
    }

    void NetworkEditor::removeSynapse(Network& network, Neuron& source, Neuron& destination)
    {
        uint32_t dest = std::stoi(destination.name);
        uint32_t src = std::stoi(source.name);

        for (auto it = std::begin(network.graph[src]->conn); it != std::end(network.graph[src]->conn); ++it)
        {
            if ((*it)->dest == dest)
            {
                network.graph[src]->conn.erase(it);
                break;
            }
        }
        for (auto it = std::begin(network.graph[dest]->sources); it != std::end(network.graph[dest]->sources); ++it)
        {
            if ((*it) == src)
            {
                network.graph[dest]->sources.erase(it);
                break;
            }
        }
    }
}
