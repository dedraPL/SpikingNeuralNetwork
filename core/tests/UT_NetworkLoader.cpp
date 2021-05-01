#include "catch.hpp"

#include <vector>

#include "network.hpp"
#include "networkLoader.hpp"
#include "node.hpp"

namespace SNN {
    TEST_CASE("SNN.NetworkLoader.load", "[Network][NetworkLoader]") {
        Network* network = new Network();
        REQUIRE_THROWS_AS(NetworkLoader::load("2net.tx", *network), NetworkLoader::FileNotFoundError);
        REQUIRE_NOTHROW(NetworkLoader::load("2net.txt", *network));
        REQUIRE(network->graph.size() == 5);

        std::vector<uint32_t> inputIdx = { 0, 1, 2 };
        std::vector<uint32_t> outputIdx = { 3, 4 };

        SECTION("connection test") {
            INFO("inner connections between nodes");

            SECTION("inputs to outputs") {
                INFO("inputs to outputs connections");
                for (uint32_t i = 0; i < 3; i++)
                {
                    REQUIRE(network->graph[i]->sources.size() == 0);
                    for (auto const& c : network->graph[i]->conn)
                    {
                        REQUIRE(
                            std::any_of(
                                outputIdx.begin(),
                                outputIdx.end(),
                                [&](const auto& x)
                                {
                                    return c->dest->name == x;
                                }
                        ));
                    }
                }
            }

            SECTION("outputs to inputs") {
                INFO("outputs to inputs back references");
                for (uint32_t i = 3; i < 5; i++)
                {
                    REQUIRE(network->graph[i]->conn.size() == 0);
                    for (auto const& c : network->graph[i]->sources)
                    {
                        REQUIRE(
                            std::any_of(
                                inputIdx.begin(),
                                inputIdx.end(),
                                [&](const auto& x)
                                {
                                    return c->src->name == x;
                                }
                        ));
                    }
                }
            }
        }
    }

    TEST_CASE("SNN.NetworkLoader.loadBin", "[Network][NetworkLoader]") {
        Network* network = new Network();
        REQUIRE_THROWS_AS(NetworkLoader::loadBin("2net.bi", *network), NetworkLoader::FileNotFoundError);
        REQUIRE_NOTHROW(NetworkLoader::loadBin("2net.bin", *network));
        REQUIRE(network->graph.size() == 5);

        std::vector<uint32_t> inputIdx = { 0, 1, 2 };
        std::vector<uint32_t> outputIdx = { 3, 4 };

        SECTION("connection test") {
            INFO("inner connections between nodes");

            SECTION("inputs to outputs") {
                INFO("inputs to outputs connections");
                for (uint32_t i = 0; i < 3; i++)
                {
                    REQUIRE(network->graph[i]->sources.size() == 0);
                    for (auto const& c : network->graph[i]->conn)
                    {
                        REQUIRE(
                            std::any_of(
                                outputIdx.begin(),
                                outputIdx.end(),
                                [&](const auto& x)
                                {
                                    return c->dest->name == x;
                                }
                        ));
                    }
                }
            }

            SECTION("outputs to inputs") {
                INFO("outputs to inputs back references");
                for (uint32_t i = 3; i < 5; i++)
                {
                    REQUIRE(network->graph[i]->conn.size() == 0);
                    for (auto const& c : network->graph[i]->sources)
                    {
                        REQUIRE(
                            std::any_of(
                                inputIdx.begin(),
                                inputIdx.end(),
                                [&](const auto& x)
                                {
                                    return c->src->name == x;
                                }
                        ));
                    }
                }
            }
        }
    }

    TEST_CASE("SNN.NetworkLoader.save", "[Network][NetworkLoader]") {
        Network* network = new Network();
        REQUIRE_NOTHROW(NetworkLoader::load("2net.txt", *network));
        REQUIRE(network->graph.size() == 5);

        std::vector<uint32_t> inputIdx = { 0, 1, 2 };
        std::vector<uint32_t> outputIdx = { 3, 4 };

        REQUIRE_THROWS_AS(NetworkLoader::save("2net_test.txt", *network), NetworkLoader::InvalidNetworkError);

        network->BFSSort();

        REQUIRE_NOTHROW(NetworkLoader::save("2net_test.txt", *network));

        Network* network_new = new Network();
        REQUIRE_NOTHROW(NetworkLoader::load("2net_test.txt", *network_new));


        REQUIRE(network->graph.size() == network_new->graph.size());
        auto graph_ref = network->graph.cbegin();
        auto graph_new = network_new->graph.cbegin();
        while(graph_ref != network->graph.cend())
        {
            auto node1 = graph_ref->second.get();
            auto node2 = graph_new->second.get();

            REQUIRE(node1->mode == node2->mode);
            REQUIRE(node1->name == node2->name);
            REQUIRE(node1->index == node2->index);

            auto synapse_ref = node1->conn.cbegin();
            auto synapse_new = node2->conn.cbegin();

            while (synapse_ref != node1->conn.cend())
            {
                auto dest1 = synapse_ref->get();
                auto dest2 = synapse_new->get();

                REQUIRE(dest1->r == dest2->r);
                REQUIRE(dest1->dest->name == dest2->dest->name);

                ++synapse_ref;
                ++synapse_new;
            }

            synapse_ref = node1->sources.cbegin();
            synapse_new = node2->sources.cbegin();

            while (synapse_ref != node1->sources.cend())
            {
                auto src1 = synapse_ref->get();
                auto src2 = synapse_new->get();

                REQUIRE(src1->r == src2->r);
                REQUIRE(src1->src->name == src2->src->name);

                ++synapse_ref;
                ++synapse_new;
            }

            ++graph_ref;
            ++graph_new;
        }

    }

    TEST_CASE("SNN.NetworkLoader.saveBin", "[Network][NetworkLoader]") {
        Network* network = new Network();
        REQUIRE_NOTHROW(NetworkLoader::loadBin("2net.bin", *network));
        REQUIRE(network->graph.size() == 5);

        std::vector<uint32_t> inputIdx = { 0, 1, 2 };
        std::vector<uint32_t> outputIdx = { 3, 4 };

        REQUIRE_THROWS_AS(NetworkLoader::saveBin("2net_test.bin", *network), NetworkLoader::InvalidNetworkError);

        network->BFSSort();

        REQUIRE_NOTHROW(NetworkLoader::saveBin("2net_test.bin", *network));

        Network* network_new = new Network();
        REQUIRE_NOTHROW(NetworkLoader::loadBin("2net_test.bin", *network_new));


        REQUIRE(network->graph.size() == network_new->graph.size());
        auto graph_ref = network->graph.cbegin();
        auto graph_new = network_new->graph.cbegin();
        while (graph_ref != network->graph.cend())
        {
            auto node1 = graph_ref->second.get();
            auto node2 = graph_new->second.get();

            REQUIRE(node1->mode == node2->mode);
            REQUIRE(node1->name == node2->name);
            REQUIRE(node1->index == node2->index);

            auto synapse_ref = node1->conn.cbegin();
            auto synapse_new = node2->conn.cbegin();

            while (synapse_ref != node1->conn.cend())
            {
                auto dest1 = synapse_ref->get();
                auto dest2 = synapse_new->get();

                REQUIRE(dest1->r == dest2->r);
                REQUIRE(dest1->dest->name == dest2->dest->name);

                ++synapse_ref;
                ++synapse_new;
            }

            synapse_ref = node1->sources.cbegin();
            synapse_new = node2->sources.cbegin();

            while (synapse_ref != node1->sources.cend())
            {
                auto src1 = synapse_ref->get();
                auto src2 = synapse_new->get();

                REQUIRE(src1->r == src2->r);
                REQUIRE(src1->src->name == src2->src->name);

                ++synapse_ref;
                ++synapse_new;
            }

            ++graph_ref;
            ++graph_new;
        }

    }
}