#pragma once

#include <string>
#include <exception>
#include <map>
#include <fstream>
#include "network.hpp"

namespace SNN {
    class NetworkLoader {
    public:
        static void load(std::string filename, Network* network);
        static void loadBin(std::string filename, Network* network);

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
    
    private:
        template<class T>
        static void loadAndProcessBinFile(std::fstream* file, uint8_t config, Network* network);
    };
}