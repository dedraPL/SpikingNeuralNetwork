#include "network.hpp"

namespace SNN { 
    Network::Network() 
    {
    }

    std::vector<std::vector<uint32_t>>* Network::BFSSort()
    {
        std::list<uint32_t> queue;
        std::vector<std::pair<uint32_t, uint32_t>> order;
        order.reserve(graph.size());

        for (auto const& [key, val] : this->graph)
        {
            if (val->mode == NodeMode::input)
            {
                queue.push_back(key);
                order.push_back({ 0, key });
                inputsIdx.push_back(key);
            }
            else if (val->mode == NodeMode::output)
            {
                outputsIdx.push_back(key);
            }
        }

        uint32_t node;
        uint32_t level = 0;
        while (!queue.empty())
        {
            node = queue.front();
            queue.pop_front();

            for (auto const& [lvl, nodeIndex] : order)
            {
                if (nodeIndex == node)
                {
                    level = lvl;
                    break;
                }
            }

            for (auto n : this->graph[node]->conn)
            {
                uint32_t r = 0;
                bool test = false;
                for (auto const& [lvl, nodeIndex] : order)
                {
                    if (nodeIndex == n->dest)
                    {
                        r = lvl;
                        test = true;
                        break;
                    }
                }
                if (test == false)
                {
                    order.push_back({ level + 1, n->dest });
                    queue.push_back(n->dest);
                }
                else if (r > level + 1)
                {
                    for (auto i : order)
                    {
                        if (i.second == r)
                        {
                            i.first = level + 1;
                            break;
                        }
                    }
                }
            }
        }

        std::sort(order.begin(), order.end(), [](auto const& l, auto const& r)
            {
                if (l.first == r.first)
                    return l.second < r.second;
                return l.first < r.first;
            });

        graphOrder.clear();
        graphOrder.resize((uint64_t)order.back().first + 1);

        for (auto const& [key, val] : order)
        {
            graphOrder[key].push_back(val);
        }

        return &graphOrder;
    }

    std::vector<double> Network::run(std::vector<double> inputs)
    {
        if (inputs.size() != inputSize)
            throw Network::InputSizeError(inputSize);

        std::vector<double> output(outputSize);

        for (auto const& layer : graphOrder)
        {
            for (auto const& nodeID : layer)
            {
                Neuron* neuron = graph[nodeID]->node;
                if (graph[nodeID]->mode == Network::NodeMode::input)
                {
                    neuron->AddCurrent(inputs[neuron->index]);
                }
                auto [v, u] = neuron->CalculatePotential();
                for (auto const& synapse : graph[nodeID]->conn)
                {
                    Node* targetNode = graph[synapse->dest];
                    targetNode->node->AddCurrent(synapse->CalculateCurrent(v, targetNode->node->prevV));
                }
                if (graph[nodeID]->mode == Network::NodeMode::output)
                {
                    output[neuron->index] = v;
                }
            }
        }
        return output;
    }
}