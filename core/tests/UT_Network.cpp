#include "catch.hpp"

#include <vector>

#include "network.hpp"
#include "networkLoader.hpp"
#include "node.hpp"

#include <Eigen/Dense>

namespace SNN {
    TEST_CASE("SNN.Network.BFSSort 2 layers", "[Network][NetworkLoader]") {
        Network* network = new Network();
        REQUIRE_NOTHROW(NetworkLoader::load("2net.txt", *network));

        network->BFSSort();

        std::vector<std::vector<uint32_t>> target = {
            {0, 1, 2},
            {3, 4} };

        for (uint32_t i = 0; i < network->graphOrder.size(); i++)
        {
            REQUIRE_THAT(network->graphOrder[i], Catch::Matchers::UnorderedEquals(target[i]));
        }
    }

    TEST_CASE("SNN.Network.BFSSort 3 layers", "[Network][NetworkLoader]") {
        Network* network = new Network();
        REQUIRE_NOTHROW(NetworkLoader::load("3net.txt", *network));

        network->BFSSort();

        std::vector<std::vector<uint32_t>> target = {
            {0, 1, 2},
            {3, 4, 5, 6},
            {7, 8, 9},
            { 10, 11 } };

        for (uint32_t i = 0; i < network->graphOrder.size(); i++)
        {
            REQUIRE_THAT(network->graphOrder[i], Catch::Matchers::UnorderedEquals(target[i]));
        }
    }

    TEST_CASE("SNN.Network.BFSSort inputs & outputs idx", "[Network][NetworkLoader]") {
        Network* network = new Network();
        REQUIRE_NOTHROW(NetworkLoader::load("3net.txt", *network));

        REQUIRE(network->getInputsIdx()->size() == 0);
        REQUIRE(network->getOutputsIdx()->size() == 0);

        network->BFSSort();
        std::vector<Node*> inputs = { network->graph[0].get(), network->graph[1].get() , network->graph[2].get() };
        std::vector<Node*> outputs = { network->graph[10].get(), network->graph[11].get() };

        REQUIRE_THAT(*network->getInputsIdx(), Catch::Matchers::UnorderedEquals(inputs));
        REQUIRE_THAT(*network->getOutputsIdx(), Catch::Matchers::UnorderedEquals(outputs));
    }

    TEST_CASE("SNN.Network.runf", "[Network][NetworkLoader]") {
        Network* network = new Network();
        REQUIRE_NOTHROW(NetworkLoader::load("2net.txt", *network));

        Eigen::Vector3f input = { 10, 10, 10 };
        
        REQUIRE_THROWS_AS(network->runf(Eigen::Vector2f::Constant(10)), Network::InputSizeError);

        network->BFSSort();

        REQUIRE_THROWS_AS(network->runf(Eigen::Vector2f::Constant(10)), Network::InputSizeError);
        REQUIRE_NOTHROW(network->runf(Eigen::Vector3f::Constant(0)));
        Eigen::Vector2f output1 = network->runf(input);
        Eigen::Vector2f output2 = network->runf(input);
        REQUIRE_THAT(output1[0], Catch::Matchers::WithinRel(-62.5));
        REQUIRE_THAT(output1[1], Catch::Matchers::WithinRel(-62.5));
        REQUIRE(((output1 - output2).norm() > 1e-6 && (output1 - output2).norm() < 2.0));
    }
}