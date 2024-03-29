#include "networkLoader.hpp"

#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <sstream>
#include "neuron.hpp"
#include "synapse.hpp"

using namespace std;

namespace SNN {
    void NetworkLoader::load(std::string filename, Network* network) 
    {
        fstream file;
        file.open(filename, ios::in);
        if(file.is_open()) {
            std::string line;
            vector<std::string> tmp;

            //read line of input nodes
            getline(file, line);
            split(tmp, line, ' ');
            vector<uint32_t> inputs(tmp.size());
            transform(tmp.begin(), tmp.end(), inputs.begin(), [](string const& val) {return stoi(val);});

            //read line of output nodes
            getline(file, line);
            split(tmp, line, ' ');
            vector<uint32_t> outputs(tmp.size());
            transform(tmp.begin(), tmp.end(), outputs.begin(), [](string const& val) {return stoi(val);});

            //load whole matrix of weights
            vector<vector<double>> matrix;
            while(getline(file, line)) 
            {
                split(tmp, line, ' ');
                matrix.push_back(vector<double>(tmp.size()));
                transform(tmp.begin(), tmp.end(), matrix.back().begin(), [](string const& val) {return stod(val);});
            }
            file.close();

            network->inputSize = inputs.size();
            network->outputSize = outputs.size();

            network->graph = map<uint32_t, Network::Node*>();
            for(uint32_t x = 0; x != matrix.size(); x++) 
            {
                Network::NodeMode mode;
                uint32_t index = 0;

                if(find(inputs.begin(), inputs.end(), x) != inputs.end()) 
                {
                    mode = Network::NodeMode::input;
                    index = std::distance(inputs.begin(), std::find(inputs.begin(), inputs.end(), x));
                    //std::cout << "i ";
                    //std::for_each(matrix[x].begin(), matrix[x].end(), [](const double n) { std::cout << n << ' '; });
                    //std::cout << '\n';
                }
                else if(find(outputs.begin(), outputs.end(), x) != outputs.end()) 
                {
                    mode = Network::NodeMode::output;
                    index = std::distance(outputs.begin(), std::find(outputs.begin(), outputs.end(), x));
                    //cout << "o ";
                    //std::for_each(matrix[x].begin(), matrix[x].end(), [](const double n) { std::cout << n << ' '; });
                    //std::cout << '\n';
                }
                else 
                {
                    mode = Network::NodeMode::hidden;
                    //std::cout << "h ";
                    //std::for_each(matrix[x].begin(), matrix[x].end(), [](const double n) { std::cout << n << ' '; });
                    //std::cout << '\n';
                }

                Neuron* node = new Neuron(to_string(x), 0.02, 0.2, -65, 8, index);
                vector<Synapse*> conn;

                for(uint32_t y = 0; y != matrix[x].size(); y++) 
                {
                    if(matrix[x][y] != 0) 
                    {
                        conn.push_back(new Synapse(y, matrix[x][y]));
                        auto it = network->graph.find(y);
                        if(it == network->graph.end()) 
                        {
                            network->graph.insert({y, new Network::Node});
                        }
                        network->graph[y]->sources.push_back(x);
                    }
                }
                auto it = network->graph.find(x);
                if(it == network->graph.end()) 
                {
                    network->graph.insert({x, new Network::Node});
                }
                network->graph[x]->update(node, mode, conn);
            }
        }
        else 
            throw NetworkLoader::FileNotFoundError(filename);
    }

    void NetworkLoader::loadBin(std::string filename, Network* network)
    {
        fstream file;
        file.open(filename, ios::in || ios::binary);
        if (file.is_open()) 
        {
            uint8_t config;
            file.read(reinterpret_cast<char*>(&config), sizeof(uint8_t));
            
            if (config & 1)
            {
                loadAndProcessBinFile<float>(&file, config, network);
            }
            else
            {
                loadAndProcessBinFile<double>(&file, config, network);
            }
            
            file.close();
        }
        else
            throw NetworkLoader::FileNotFoundError(filename);
    }

    void NetworkLoader::save(std::string filename, Network* network)
    {
        if (network->graph.size() > 0 && network->getInputsIdx().size() > 0 && network->getOutputsIdx().size() > 0)
        {
            fstream file;
            file.open(filename, ios::out);

            for (auto const& val : network->getInputsIdx())
            {
                file << val;
                if (&val != &network->getInputsIdx().back())
                    file << " ";
            }
            file << std::endl;
            for (auto const& val : network->getOutputsIdx())
            {
                file << val;
                if (&val != &network->getOutputsIdx().back())
                    file << " ";
            }
            file << std::endl;

            uint32_t size = network->graph.size();
            for (auto const& [key, val] : network->graph)
            {
                std::vector<Synapse*> conn = val->getConn();
                std::sort(std::begin(conn), std::end(conn), [](Synapse* a, Synapse* b)
                    {
                        return a->dest < b->dest;
                    });
                std::vector<Synapse*>::iterator conn_val = conn.begin();
                for (uint32_t i = 0; i < size; i++)
                {
                    if (conn.size() > 0 && (*conn_val)->dest == i)
                    {
                        file << (*conn_val)->r;
                        if (conn_val + 1 != conn.end())
                            ++conn_val;
                    }
                    else
                        file << 0;
                    if (i + 1 < size)
                        file << ' ';
                }
                file << std::endl;
            }
            file.close();
        }
        else
            throw NetworkLoader::InvalidNetworkError();
    }

    void NetworkLoader::saveBin(std::string filename, Network* network)
    {
        if (network->graph.size() > 0 && network->getInputsIdx().size() > 0 && network->getOutputsIdx().size() > 0)
        {
            fstream file;
            file.open(filename, ios::out | ios::binary);

            uint8_t config = 0;
            config |= typeid(Synapse::r) == typeid(float);
            file.write(reinterpret_cast<const char*>(&config), sizeof(config));
            uint32_t size = network->graph.size();
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
            uint32_t io_size = network->getInputsIdx().size();
            file.write(reinterpret_cast<const char*>(&io_size), sizeof(io_size));
            io_size = network->getOutputsIdx().size();
            file.write(reinterpret_cast<const char*>(&io_size), sizeof(io_size));

            std::vector<uint32_t> inputs, outputs;
            inputs = network->getInputsIdx();
            outputs = network->getOutputsIdx();

            file.write((char*)&inputs[0], inputs.size() * sizeof(uint32_t));
            file.write((char*)&outputs[0], outputs.size() * sizeof(uint32_t));

            if (config & 1)
                processAndSaveBinFile<float>(&file, network);
            else
                processAndSaveBinFile<double>(&file, network);

            file.close();
        }
        else
            throw NetworkLoader::InvalidNetworkError();
    }

    template<class T>
    void NetworkLoader::loadAndProcessBinFile(std::fstream* file, uint8_t config, Network* network)
    {
        uint32_t networkSize;
        uint32_t inputsSize;
        uint32_t outputsSize;

        file->read(reinterpret_cast<char*>(&networkSize), sizeof(uint32_t));
        file->read(reinterpret_cast<char*>(&inputsSize), sizeof(uint32_t));
        file->read(reinterpret_cast<char*>(&outputsSize), sizeof(uint32_t));
        std::vector<uint32_t> inputs(inputsSize);
        std::vector<uint32_t> outputs(outputsSize);
        file->read((char*)&inputs[0], inputsSize * sizeof(uint32_t));
        file->read((char*)&outputs[0], outputsSize * sizeof(uint32_t));

        std::vector<std::vector<T>> matrix(networkSize);

        for (uint32_t i = 0; i < networkSize; i++)
        {
            matrix[i].resize(networkSize);
            file->read(reinterpret_cast<char*>(&matrix[i][0]), networkSize * sizeof(T));
        }

        network->inputSize = inputs.size();
        network->outputSize = outputs.size();

        network->graph = map<uint32_t, Network::Node*>();
        for (uint32_t x = 0; x != matrix.size(); x++)
        {
            Network::NodeMode mode;
            uint32_t index = 0;

            if (find(inputs.begin(), inputs.end(), x) != inputs.end())
            {
                mode = Network::NodeMode::input;
                index = std::distance(inputs.begin(), std::find(inputs.begin(), inputs.end(), x));
                //std::cout << "i ";
                //std::for_each(matrix[x].begin(), matrix[x].end(), [](const double n) { std::cout << n << ' '; });
                //std::cout << '\n';
            }
            else if (find(outputs.begin(), outputs.end(), x) != outputs.end())
            {
                mode = Network::NodeMode::output;
                index = std::distance(outputs.begin(), std::find(outputs.begin(), outputs.end(), x));
                //cout << "o ";
                //std::for_each(matrix[x].begin(), matrix[x].end(), [](const double n) { std::cout << n << ' '; });
                //std::cout << '\n';
            }
            else
            {
                mode = Network::NodeMode::hidden;
                //std::cout << "h ";
                //std::for_each(matrix[x].begin(), matrix[x].end(), [](const double n) { std::cout << n << ' '; });
                //std::cout << '\n';
            }

            Neuron* node = new Neuron(to_string(x), 0.02, 0.2, -65, 8, index);
            vector<Synapse*> conn;

            for (uint32_t y = 0; y != matrix[x].size(); y++)
            {
                if (matrix[x][y] != 0)
                {
                    conn.push_back(new Synapse(y, matrix[x][y]));
                    auto it = network->graph.find(y);
                    if (it == network->graph.end())
                    {
                        network->graph.insert({ y, new Network::Node });
                    }
                    network->graph[y]->sources.push_back(x);
                }
            }
            auto it = network->graph.find(x);
            if (it == network->graph.end())
            {
                network->graph.insert({ x, new Network::Node });
            }
            network->graph[x]->update(node, mode, conn);
        }
    }

    template<class T>
    void NetworkLoader::processAndSaveBinFile(std::fstream* file, Network* network)
    {
        uint32_t size = network->graph.size();
        T* buffer = new T[size];
        for (auto const& [key, val] : network->graph)
        {
            std::vector<Synapse*> conn = val->getConn();
            std::sort(std::begin(conn), std::end(conn), [](Synapse* a, Synapse* b)
                {
                    return a->dest < b->dest;
                });
            std::vector<Synapse*>::iterator conn_val = conn.begin();
            for (uint32_t i = 0; i < size; i++)
            {
                if (conn.size() > 0 && (*conn_val)->dest == i)
                {
                    buffer[i] = (*conn_val)->r;
                    if (conn_val + 1 != conn.end())
                        ++conn_val;
                }
                else
                    buffer[i] = 0;
            }
            file->write(reinterpret_cast<const char*>(buffer), sizeof(T) * size);
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