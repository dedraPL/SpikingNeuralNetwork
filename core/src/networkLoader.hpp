#pragma once

#include <string>
#include <exception>
#include <map>
#include <fstream>
#include <nlohmann/json.hpp>
#include "network.hpp"
#include "neuron.hpp"

namespace SNN {
    class NetworkLoader {
    public:
        static void load(std::string filename, Network& network);
        static void loadBin(std::string filename, Network& network);

        /// @brief Saves the network to the file. Requires to run Network::BFSSort before saving
        /// @param filename 
        /// @param network Network to save
        static void save(std::string filename, Network& network);

        /// @brief Saves the network to the binary file. Requires to run Network::BFSSort before saving
        /// @param filename 
        /// @param network Network to save
        static void saveBin(std::string filename, Network& network);

        class FileNotFoundError : public std::exception {
            std::string _msg;
        public:
            explicit FileNotFoundError(const std::string& msg)
            {
                _msg = std::string("Unable to load file ") + _msg;
            }
            const char * what () const throw () {
                return _msg.c_str();
            }
        };

        class InvalidNetworkError : public std::exception {
            std::string _msg;
        public:
            explicit InvalidNetworkError()
            {
                _msg = "Network is not sorted or is empty";
            }
            const char* what() const throw () {
                return _msg.c_str();
            }
        };

        struct Model {
            NEURON_TYPE a;
            NEURON_TYPE b;
            NEURON_TYPE c;
            NEURON_TYPE d;
        };
        
    private:
        template<class T>
        static void loadAndProcessBinFile(nlohmann::json& j, Network& network);
        template<class T>
        static void processAndSaveBinFile(std::string filename, Network& network);
        
        static std::vector<std::string>& split(std::vector<std::string>& result, const std::string& input, char delimiter);

        static bool approximatelyEqual(NEURON_TYPE a, NEURON_TYPE b, NEURON_TYPE epsilon)
        {
            return std::abs(a - b) <= ((std::abs(a) < std::abs(b) ? std::abs(b) : std::abs(a)) * epsilon);
        }
    };

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NetworkLoader::Model, a, b, c, d)
}