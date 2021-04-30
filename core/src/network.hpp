#pragma once

#include <string>
#include <map>
#include <vector>
#include <list>
#include <exception>
#include <algorithm>

#include <Eigen/Dense>

#include "neuron.hpp"
#include "synapse.hpp"
#include "node.hpp"

namespace SNN {
    class Network {
    public:
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
        ~Network();
        std::vector<std::vector<uint32_t>>* BFSSort();
        std::vector<double> run(std::vector<double> inputs);
        Eigen::VectorXd rund(const Eigen::Ref<const Eigen::VectorXd>& inputs);
        Eigen::VectorXf runf(const Eigen::Ref<const Eigen::VectorXf>& inputs);

        std::map<uint32_t, std::shared_ptr<Node>> graph;
        std::map<uint32_t, std::shared_ptr<Node>>* getGraph() { return &graph; };
        std::vector<Node*>* getInputsIdx() { return &inputsIdx; };
        std::vector<Node*>* getOutputsIdx() { return &outputsIdx; };
        uint32_t inputSize = 0, outputSize = 0;
        std::vector<std::vector<uint32_t>> graphOrder;
    private:
        std::vector<Node*> inputsIdx, outputsIdx;
        template<class T>
        T run(const Eigen::Ref<const T>& inputs);
    };

    template<typename T>
    inline T Network::run(const Eigen::Ref<const T>& inputs)
    {
        if (inputs.rows() != inputSize && inputs.cols() != 1)
            throw Network::InputSizeError(inputSize);

        T output = T(outputSize);

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
}