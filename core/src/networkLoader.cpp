#include "networkLoader.hpp"

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
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
            boost::split(tmp, line, boost::is_any_of(" "), boost::token_compress_on);
            vector<int> inputs(tmp.size());
            transform(tmp.begin(), tmp.end(), inputs.begin(), [](string const& val) {return stoi(val);});

            //read line of output nodes
            getline(file, line);
            boost::split(tmp, line, boost::is_any_of(" "), boost::token_compress_on);
            vector<int> outputs(tmp.size());
            transform(tmp.begin(), tmp.end(), outputs.begin(), [](string const& val) {return stoi(val);});

            //load whole matrix of weights
            vector<vector<double>> matrix;
            while(getline(file, line)) 
            {
                boost::split(tmp, line, boost::is_any_of(" "), boost::token_compress_on);
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
}