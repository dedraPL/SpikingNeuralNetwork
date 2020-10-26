#include "networkLoader.hpp"

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include "neuron.hpp"
#include "synapse.hpp"

using namespace std;

namespace SNN {
    void NetworkLoader::load(std::string filename, Network* network) {
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
            while(getline(file, line)) {
                boost::split(tmp, line, boost::is_any_of(" "), boost::token_compress_on);
                matrix.push_back(vector<double>(tmp.size()));
                transform(tmp.begin(), tmp.end(), matrix.back().begin(), [](string const& val) {return stod(val);});
            }
            file.close();

            network->inputSize = inputs.size();
            network->outputSize = outputs.size();

            network->graph = map<uint32_t, Network::Node*>();
            for(uint32_t x = 0; x != matrix.size(); x++) {
                Network::NodeMode mode;
                uint32_t index = 0;

                if(find(inputs.begin(), inputs.end(), x) != inputs.end()) {
                    mode = Network::NodeMode::input;
                    index = std::distance(inputs.begin(), std::find(inputs.begin(), inputs.end(), x));
                    //std::cout << "i ";
                    //std::for_each(matrix[x].begin(), matrix[x].end(), [](const double n) { std::cout << n << ' '; });
                    //std::cout << '\n';
                }
                else if(find(outputs.begin(), outputs.end(), x) != outputs.end()) {
                    mode = Network::NodeMode::output;
                    index = std::distance(outputs.begin(), std::find(outputs.begin(), outputs.end(), x));
                    //cout << "o ";
                    //std::for_each(matrix[x].begin(), matrix[x].end(), [](const double n) { std::cout << n << ' '; });
                    //std::cout << '\n';
                }
                else {
                    mode = Network::NodeMode::hidden;
                    //std::cout << "h ";
                    //std::for_each(matrix[x].begin(), matrix[x].end(), [](const double n) { std::cout << n << ' '; });
                    //std::cout << '\n';
                }

                Neuron* node = new Neuron(to_string(x), 0.02, 0.2, -65, 8, index);
                vector<Synapse*> conn;

                for(uint32_t y = 0; y != matrix[x].size(); y++) {
                    if(matrix[x][y] != 0) {
                        conn.push_back(new Synapse(y, matrix[x][y]));
                        auto it = network->graph.find(y);
                        if(it == network->graph.end()) {
                            network->graph.insert({y, new Network::Node});
                        }
                        network->graph[y]->sources.push_back(x);
                    }
                }
                auto it = network->graph.find(x);
                if(it == network->graph.end()) {
                    network->graph.insert({x, new Network::Node});
                }
                network->graph[x]->update(node, mode, conn);
            }
        }
        else 
            throw NetworkLoader::FileNotFoundError(filename);
    }
}