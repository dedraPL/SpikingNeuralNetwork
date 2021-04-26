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

namespace SNN {
    class Network {
    public:
        enum class NodeMode { input, output, hidden };
        
        struct Node {
            Neuron* node;
            NodeMode mode;
            std::vector<Synapse*> conn;
            std::vector<uint32_t> sources;

            void update(Neuron& node, NodeMode mode, std::vector<Synapse*> conn) {
                this->node = &node;
                this->mode = mode;
                this->conn = conn;
            }

            Node() {}

            ~Node()
            {
                for (auto it = std::begin(conn); it != std::end(conn); ++it)
                {
                    delete(*it);
                }
                delete(node);
            }

            Neuron* getNode() { return node; }
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
        Eigen::VectorXd rund(const Eigen::Ref<const Eigen::VectorXd>& inputs);
        Eigen::VectorXf runf(const Eigen::Ref<const Eigen::VectorXf>& inputs);

        std::map<uint32_t, Network::Node*> graph;
        std::map<uint32_t, Network::Node*>* getGraph() { return &graph; };
        std::vector<uint32_t>* getInputsIdx() { return &inputsIdx; };
        std::vector<uint32_t>* getOutputsIdx() { return &outputsIdx; };
        uint32_t inputSize = 0, outputSize = 0;
        std::vector<std::vector<uint32_t>> graphOrder;
    private:
        std::vector<uint32_t> inputsIdx, outputsIdx;
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