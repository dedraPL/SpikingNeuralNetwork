#pragma once

#include <string>
#include <exception>
#include <map>
#include <fstream>
#include "network.hpp"

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
    
    private:
        template<class T>
        static void loadAndProcessBinFile(std::fstream* file, uint8_t config, Network* network);
        template<class T>
        static void processAndSaveBinFile(std::fstream* file, Network* network);
        
        static std::vector<std::string>& split(std::vector<std::string>& result, const std::string& input, char delimiter);
    };
}