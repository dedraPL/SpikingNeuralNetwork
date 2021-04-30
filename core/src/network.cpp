#include "network.hpp"

namespace SNN { 
    Network::Network() 
    {
    }

    Network::~Network()
    {
    }

    std::vector<std::vector<uint32_t>>* Network::BFSSort()
    {
        std::list<uint32_t> queue;
        std::vector<std::pair<uint32_t, uint32_t>> order;
        order.reserve(graph.size());

        inputsIdx.clear();
        outputsIdx.clear();
        uint32_t name = 0;
        std::map<uint32_t, std::shared_ptr<Node>> tmpMap;
        for (auto it = this->graph.begin(); it != this->graph.end();)
        {
            auto tmp = this->graph.extract(it++);
            tmp.key() = name;
            auto tmp2 = tmpMap.insert(tmpMap.end(), std::move(tmp));
            tmp2->second->name = name;

            if ((*tmp2).second->mode == Node::NodeMode::input)
            {
                queue.push_back(name);
                order.push_back({ 0, name });
                inputsIdx.push_back((*tmp2).second.get());
            }
            else if ((*tmp2).second->mode == Node::NodeMode::output)
            {
                outputsIdx.push_back((*tmp2).second.get());
            }

            name++;
        }
        tmpMap.swap(this->graph);

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
                    if (nodeIndex == n->dest->name)
                    {
                        r = lvl;
                        test = true;
                        break;
                    }
                }
                if (test == false)
                {
                    order.push_back({ level + 1, n->dest->name });
                    queue.push_back(n->dest->name);
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
                /*Neuron* neuron = graph[nodeID]->node.get();
                if (graph[nodeID]->mode == Network::NodeMode::input)
                {
                    neuron->AddCurrent(inputs[neuron->index]);
                }
                auto [v, u] = neuron->CalculatePotential();
                for (auto const& synapse : graph[nodeID]->conn)
                {
                    Neuron* targetNode = graph[synapse->dest->index].get()->node.get();
                    targetNode->AddCurrent(synapse->CalculateCurrent(v, targetNode->prevV));
                }
                if (graph[nodeID]->mode == Network::NodeMode::output)
                {
                    output[neuron->index] = v;
                }*/
                Node* node = graph[nodeID].get();
                if (node->mode == Node::NodeMode::input)
                {
                    node->node->AddCurrent(inputs[node->index]);
                }
                auto [v, u] = node->node->CalculatePotential();
                for (auto const& synapse : node->conn)
                {
                    Neuron* targetNode = synapse->dest->node.get();
                    targetNode->AddCurrent(synapse->CalculateCurrent(v, targetNode->prevV));
                }
                if (node->mode == Node::NodeMode::output)
                {
                    output[node->index] = v;
                }
            }
        }
        return output;
    }

    Eigen::VectorXd Network::rund(const Eigen::Ref<const Eigen::VectorXd>& inputs)
    {
        return this->run<Eigen::VectorXd>(inputs);
    }

    Eigen::VectorXf Network::runf(const Eigen::Ref<const Eigen::VectorXf>& inputs)
    {
        return this->run<Eigen::VectorXf>(inputs);
    }
}