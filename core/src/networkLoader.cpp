#include "networkLoader.hpp"

#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <sstream>
#include <iomanip>
#include "synapse.hpp"

#define FORMAT "format"
#define INPUTS "inputs"
#define OUTPUTS "outputs"
#define MATRIX "matrix"
#define DESCRIPTION "description"
#define SYNAPSE_C "synapse_c"
#define MODELS "models"
#define NETWORK_SIZE "network_size"
#define NODES "nodes"

using namespace std;

namespace SNN {
    void NetworkLoader::load(std::string filename, Network& network) 
    {
        std::fstream file;
        file.open(filename, ios::in);
        if(file.is_open()) {
            nlohmann::json j;
            file >> j;
            file.close();

            std::vector<uint32_t> inputs;
            j[INPUTS].get_to(inputs);

            std::vector<uint32_t> outputs;
            j[OUTPUTS].get_to(outputs);

            if (j.contains(SYNAPSE_C))
            {
                network.setSynapseC(j[SYNAPSE_C]);
            }

            std::vector<Model> models;
            models.push_back({0.02, 0.2, -65, 8});
            std::vector<uint32_t> nodes;
            if (j.contains(MODELS))
            {
                for (auto const& model : j[MODELS].at(MODELS))
                {
                    models.push_back(model.get<NetworkLoader::Model>());
                }
                j[MODELS].at(NODES).get_to(nodes);
            }
            
            std::vector<std::vector<double>> matrix;
            for (auto const& line : j[MATRIX])
            {
                std::stringstream stream((std::string)line);
                matrix.push_back(std::vector<double>( 
                    (std::istream_iterator<double>(stream)),
                    istream_iterator<double>()
                    ));
            }

            network.inputSize = inputs.size();
            network.outputSize = outputs.size();

            network.graph = map<uint32_t, std::shared_ptr<Node>>();
            for(uint32_t x = 0; x != matrix.size(); x++) 
            {
                Node::NodeMode mode;
                uint32_t index = 0;

                if(find(inputs.begin(), inputs.end(), x) != inputs.end()) 
                {
                    mode = Node::NodeMode::input;
                    index = std::distance(inputs.begin(), std::find(inputs.begin(), inputs.end(), x));
                }
                else if(find(outputs.begin(), outputs.end(), x) != outputs.end()) 
                {
                    mode = Node::NodeMode::output;
                    index = std::distance(outputs.begin(), std::find(outputs.begin(), outputs.end(), x));
                }
                else 
                {
                    mode = Node::NodeMode::hidden;
                }

                Node* node;
                auto it = network.graph.find(x);
                if (it == network.graph.end() || (*it).second->node->a == 0)
                {
                    if (it == network.graph.end())
                        node = network.graph.insert(std::make_pair(x, std::make_shared<Node>(x, index))).first->second.get();
                    else
                        node = (*it).second.get();
                    Model m;
                    if (nodes.size() == matrix.size())
                        m = models[nodes[x]];
                    else
                        m = models[0];
                    node->node->a = m.a;
                    node->node->b = m.b;
                    node->node->c = m.c;
                    node->node->d = m.d;
                    node->mode = mode;
                    node->index = index;
                }
                else
                {
                    node = (*it).second.get();
                }

                for(uint32_t y = 0; y != matrix[x].size(); y++) 
                {
                    if(matrix[x][y] != 0) 
                    {
                        auto it = network.graph.find(y);
                        if(it == network.graph.end()) 
                        {
                            network.graph.insert(std::make_pair(y, std::make_shared<Node>(y, 0)));
                            it = network.graph.find(y);
                        }
                        node->conn.push_back(std::make_shared<Synapse>(node, it->second.get(), matrix[x][y]));
                        node->conn.back()->C = network.synapse_c;
                        network.graph[y]->sources.push_back(node->conn.back());
                    }
                }
            }
        }
        else 
            throw NetworkLoader::FileNotFoundError(filename);
    }

    void NetworkLoader::loadBin(std::string filename, Network& network)
    {
        fstream file;
        file.open(filename, ios::in || ios::binary);
        if (file.is_open()) 
        {
            nlohmann::json j = nlohmann::json::from_msgpack(file);
            file.close();

            if (j[FORMAT] == "f")
            {
                loadAndProcessBinFile<float>(j, network);
            }
            else
            {
                loadAndProcessBinFile<double>(j, network);
            }
        }
        else
            throw NetworkLoader::FileNotFoundError(filename);
    }

    void NetworkLoader::save(std::string filename, Network& network)
    {
        if (network.graph.size() > 0 && network.getInputsIdx()->size() > 0 && network.getOutputsIdx()->size() > 0)
        {
            fstream file;

            nlohmann::json j;
            j[FORMAT] = typeid(Synapse::r) == typeid(float) ? "f" : "d";
            j[SYNAPSE_C] = network.getSynapseC();
            j[NETWORK_SIZE] = network.graph.size();
            j[INPUTS] = nlohmann::json::array();
            j[OUTPUTS] = nlohmann::json::array();
            for (auto const& val : *network.getInputsIdx())
            {
                j[INPUTS].push_back(val->name);
            }
            for (auto const& val : *network.getOutputsIdx())
            {
                j[OUTPUTS].push_back(val->name);
            }

            
            j[MATRIX] = nlohmann::json::array();
            uint32_t size = network.graph.size();
            std::vector<uint32_t> nodes;
            std::vector<Model> models;
            models.push_back({ 0.02, 0.2, -65, 8 });
            for (auto const& [key, val] : network.graph)
            {
                std::stringstream matrix;
                std::vector<std::shared_ptr<Synapse>> conn = val->getConn();
                std::sort(std::begin(conn), std::end(conn), [](std::shared_ptr<Synapse> a, std::shared_ptr<Synapse> b)
                    {
                        return a->dest->name < b->dest->name;
                    });
                std::vector<std::shared_ptr<Synapse>>::iterator conn_val = conn.begin();

                for (auto it = network.graph.begin(); it != network.graph.end(); ++it)
                {
                    if (conn.size() > 0 && (*conn_val)->dest == it->second.get())
                    {
                        matrix << (*conn_val)->r;
                        if (conn_val + 1 != conn.end())
                            ++conn_val;
                    }
                    else
                        matrix << 0;
                    if (it != network.graph.end()--)
                        matrix << ' ';
                }
                auto node = val->node;
                auto model = std::find_if(models.begin(), models.end(), [node](Model m) { 
                    return approximatelyEqual(node->a, m.a, 0.0001) && approximatelyEqual(node->b, m.b, 0.0001) && approximatelyEqual(node->c, m.c, 0.0001) && approximatelyEqual(node->d, m.d, 0.0001);
                    });
                uint32_t index;
                if (model == models.end())
                {
                    models.push_back({ node->a, node->b, node->c, node->d });
                    index = models.size() - 1;
                }
                else
                {
                    index = std::distance(models.begin(), model);
                }
                nodes.push_back(index);
                j[MATRIX].push_back(matrix.str());
            }
            if (models.size() > 1)
            {
                j[MODELS] = nlohmann::json();
                j[MODELS][MODELS] = models;
                j[MODELS][NODES] = nodes;
            }

            file.open(filename, ios::out);
            file << std::setw(4) << j << std::endl;
            file.close();
        }
        else
            throw NetworkLoader::InvalidNetworkError();
    }

    void NetworkLoader::saveBin(std::string filename, Network& network)
    {
        if (network.graph.size() > 0 && network.getInputsIdx()->size() > 0 && network.getOutputsIdx()->size() > 0)
        {
            if (typeid(Synapse::r) == typeid(float))
                processAndSaveBinFile<float>(filename, network);
            else
                processAndSaveBinFile<double>(filename, network);
        }
        else
            throw NetworkLoader::InvalidNetworkError();
    }

    template<class T>
    void NetworkLoader::loadAndProcessBinFile(nlohmann::json& j, Network& network)
    {
        std::vector<uint32_t> inputs;
        j[INPUTS].get_to(inputs);

        std::vector<uint32_t> outputs;
        j[OUTPUTS].get_to(outputs);

        if (j.contains(SYNAPSE_C))
        {
            network.setSynapseC(j[SYNAPSE_C]);
        }

        std::vector<Model> models;
        models.push_back({ 0.02, 0.2, -65, 8 });
        std::vector<uint32_t> nodes;
        if (j.contains(MODELS))
        {
            for (auto const& model : j[MODELS].at(MODELS))
            {
                models.push_back(model.get<NetworkLoader::Model>());
            }
            j[MODELS].at(NODES).get_to(nodes);
        }

        uint32_t networkSize = j[NETWORK_SIZE];
        std::vector<std::vector<T>> matrix(networkSize);
        auto& bin = j[MATRIX].get_binary();
        for (uint32_t i = 0; i < networkSize; i++)
        {
            matrix[i].resize(networkSize);
            std::memcpy(&matrix[i][0], &bin[i * networkSize * sizeof(T)], networkSize * sizeof(T));
        }

        network.inputSize = inputs.size();
        network.outputSize = outputs.size();

        network.graph = map<uint32_t, std::shared_ptr<Node>>();
        for (uint32_t x = 0; x != matrix.size(); x++)
        {
            Node::NodeMode mode;
            uint32_t index = 0;

            if (find(inputs.begin(), inputs.end(), x) != inputs.end())
            {
                mode = Node::NodeMode::input;
                index = std::distance(inputs.begin(), std::find(inputs.begin(), inputs.end(), x));
            }
            else if (find(outputs.begin(), outputs.end(), x) != outputs.end())
            {
                mode = Node::NodeMode::output;
                index = std::distance(outputs.begin(), std::find(outputs.begin(), outputs.end(), x));
            }
            else
            {
                mode = Node::NodeMode::hidden;
            }

            Node* node;
            auto it = network.graph.find(x);
            if (it == network.graph.end() || (*it).second->node->a == 0)
            {
                if (it == network.graph.end())
                    node = network.graph.insert(std::make_pair(x, std::make_shared<Node>(x, index))).first->second.get();
                else
                    node = (*it).second.get();

                Model m;
                if (nodes.size() == matrix.size())
                    m = models[nodes[x]];
                else
                    m = models[0];
                node->node->a = m.a;
                node->node->b = m.b;
                node->node->c = m.c;
                node->node->d = m.d;
                node->mode = mode;
                node->index = index;
            }
            else
            {
                node = (*it).second.get();
            }

            for (uint32_t y = 0; y != matrix[x].size(); y++)
            {
                if (matrix[x][y] != 0)
                {
                    auto it = network.graph.find(y);
                    if (it == network.graph.end())
                    {
                        network.graph.insert(std::make_pair(y, std::make_shared<Node>(y, 0)));
                        it = network.graph.find(y);
                    }
                    node->conn.push_back(std::make_shared<Synapse>(node, it->second.get(), matrix[x][y]));
                    network.graph[y]->sources.push_back(node->conn.back());
                }
            }
        }
    }

    template<class T>
    void NetworkLoader::processAndSaveBinFile(std::string filename, Network& network)
    {
        nlohmann::json j;
        j[FORMAT] = typeid(T) == typeid(float) ? "f" : "d";
        j[SYNAPSE_C] = (T)network.getSynapseC();
        j[NETWORK_SIZE] = network.graph.size();
        j[INPUTS] = nlohmann::json::array();
        j[OUTPUTS] = nlohmann::json::array();
        for (auto const& val : *network.getInputsIdx())
        {
            j[INPUTS].push_back(val->name);
        }
        for (auto const& val : *network.getOutputsIdx())
        {
            j[OUTPUTS].push_back(val->name);
        }

        std::vector<uint32_t> nodes;
        std::vector<Model> models;
        uint32_t size = network.graph.size();
        std::vector<uint8_t> buffer;
        buffer.reserve(size * size * sizeof(T));
        for (auto const& [key, val] : network.graph)
        {
            std::vector<std::shared_ptr<Synapse>> conn = val->getConn();
            std::sort(conn.begin(), conn.end(), [](const std::shared_ptr<Synapse>& a, const std::shared_ptr<Synapse>& b)
                {
                    return a->dest->name < b->dest->name;
                });
            std::vector<std::shared_ptr<Synapse>>::iterator conn_val = conn.begin();
            for (uint32_t i = 0; i < size; i++)
            {
                float data;
                if (conn.size() > 0 && (*conn_val)->dest->name == i)
                {
                    data = (*conn_val)->r;
                    if (conn_val + 1 != conn.end())
                        ++conn_val;
                }
                else
                    data = 0;
                uint8_t bytes[sizeof(T)];
                std::memcpy(bytes, &data, sizeof(T));
                buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
            }
            auto node = val->node;
            auto model = std::find_if(models.begin(), models.end(), [node](Model m) {
                return approximatelyEqual(node->a, m.a, 0.0001) && approximatelyEqual(node->b, m.b, 0.0001) && approximatelyEqual(node->c, m.c, 0.0001) && approximatelyEqual(node->d, m.d, 0.0001);
                });
            uint32_t index;
            if (model == models.end())
            {
                models.push_back({ node->a, node->b, node->c, node->d });
                index = models.size() - 1;
            }
            else
            {
                index = std::distance(models.begin(), model);
            }
            nodes.push_back(index);
        }
        if (models.size() > 1)
        {
            j[MODELS] = nlohmann::json();
            models.erase(models.begin());
            j[MODELS][MODELS] = models;
            j[MODELS][NODES] = nodes;
        }

        j[MATRIX] = nlohmann::json::binary(buffer);

        fstream file;
        file.open(filename, ios::out | ios::binary);
        if (file.is_open())
        {
            std::vector<uint8_t> bin = j.to_msgpack(j);
            file.write((char*)&bin[0], bin.size() * sizeof(uint8_t));
            file.close();
        }
    }

    std::vector<std::string>& NetworkLoader::split(std::vector<std::string>& result, const std::string& input, char delimiter)
    {
        std::string token;
        std::istringstream tokenStream(input);
        result.clear();
        while (std::getline(tokenStream, token, delimiter))
        {
            result.push_back(token);
        }
        return result;
    }
}