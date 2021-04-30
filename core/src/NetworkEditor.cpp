#include <vector>

#include "NetworkEditor.hpp"

namespace SNN {
    std::shared_ptr<Node> NetworkEditor::addHiddenNode(Network& network)
    {
        return addNode(network, 0, Node::NodeMode::hidden);
    }

    std::shared_ptr<Node> NetworkEditor::addNode(Network& network, uint32_t index, Node::NodeMode mode)
    {
        uint32_t name = network.graph.size();

        std::shared_ptr<Node> node = network.graph.insert(std::make_pair(name, std::make_shared<Node>(name, index))).first->second;
        node->node->a = 0.02;
        node->node->b = 0.2;
        node->node->c = -65;
        node->node->d = 8;
        node->mode = mode;

        if (mode == Node::NodeMode::input) 
        {
            network.inputSize++;
        }
        else if (mode == Node::NodeMode::output)
        {
            network.outputSize++;
        }

        return node;
    }

    std::shared_ptr<Synapse> NetworkEditor::addSynapse(Network& network, Node& source, Node& destination, SYNAPSE_TYPE r)
    {
        uint32_t dest = destination.name;
        uint32_t src = source.name;
        std::shared_ptr<Synapse> synapse = std::make_shared<Synapse>(&source, &destination, r);
        network.graph[src]->conn.push_back(synapse);
        network.graph[dest]->sources.push_back(synapse);
        return synapse;
    }

    void NetworkEditor::removeNode(Network& network, Node& node)
    {
        {
            auto input = std::begin(node.sources);
            //uint32_t dest = node.name;
            while (input != std::end(node.sources))
            {
                /*uint32_t src = (*input)->src->name;

                for (auto it = std::begin(network.graph[src]->conn); it != std::end(network.graph[src]->conn); ++it)
                {
                    if ((*it)->dest == node.node.get())
                    {
                        network.graph[src]->conn.erase(it);
                        break;
                    }
                }*/
                for (auto it = std::begin((*input)->src->conn); it != std::end((*input)->src->conn); ++it)
                {
                    if ((*it)->dest == &node)
                    {
                        (*input)->src->conn.erase(it);
                        break;
                    }
                }
                input = node.sources.erase(input);
            }
        }

        {
            auto output = std::begin(node.conn);
            uint32_t src = node.name;
            while (output != std::end(node.conn))
            {
                /*uint32_t dest = (*output)->dest->name;

                output = node.conn.erase(output);

                for (auto it = std::begin(network.graph[dest]->sources); it != std::end(network.graph[dest]->sources); ++it)
                {
                    if ((*it)->src == node.node.get())
                    {
                        network.graph[dest]->sources.erase(it);
                        break;
                    }
                }*/
                
                for (auto it = std::begin((*output)->src->sources); it != std::end((*output)->src->sources); ++it)
                {
                    if ((*it)->src == &node)
                    {
                        (*output)->src->sources.erase(it);
                        break;
                    }
                }
                output = node.conn.erase(output);
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

    void NetworkEditor::removeSynapse(Network& network, Node& source, Node& destination)
    {
        uint32_t dest = destination.name;
        uint32_t src = source.name;
        
        for (auto it = std::begin(source.conn); it != std::end(source.conn); ++it)
        {
            if ((*it)->dest == &destination)
            {
                source.conn.erase(it);
                break;
            }
        }
        for (auto it = std::begin(destination.sources); it != std::end(destination.sources); ++it)
        {
            if ((*it)->src == &source)
            {
                destination.sources.erase(it);
                break;
            }
        }
    }
}
