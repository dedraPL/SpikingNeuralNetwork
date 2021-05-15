#include "catch.hpp"

#include <vector>

#include "network.hpp"
#include "networkEditor.hpp"
#include "networkLoader.hpp"
#include "node.hpp"
#include "synapse.hpp"

#define REQUIRE_THAT_FALSE( arg, matcher ) INTERNAL_CHECK_THAT( "REQUIRE_THAT", matcher, Catch::ResultDisposition::Normal | Catch::ResultDisposition::FalseTest, arg )

namespace SNN {
    TEST_CASE("SNN.NetworkEditor.addNode", "[Network][NetworkEditor]") {
        std::vector<uint32_t> inputNames = { 0, 1, 2 };
        std::vector<uint32_t> outputNames = { 3, 4 };
        std::map<uint32_t, uint32_t> inputIndexes = { {0, 0}, {1, 1}, {2, 2} };
        std::map<uint32_t, uint32_t> outputIndexes = { {3, 0}, {4, 1} };

        Network* network = new Network();

        uint32_t netSize = 0;
        for (auto const& input : inputIndexes)
        {
            auto node = NetworkEditor::addNode(*network, input.second, Node::NodeMode::input);
            netSize++;
            REQUIRE(network->graph.size() == netSize);
            REQUIRE(node->mode == Node::NodeMode::input);
        }

        for (auto const& input : outputIndexes)
        {
            auto node = NetworkEditor::addNode(*network, input.second, Node::NodeMode::output);
            netSize++;
            REQUIRE(network->graph.size() == netSize);
            REQUIRE(node->mode == Node::NodeMode::output);
        }

        REQUIRE(network->inputSize == 3);
        REQUIRE(network->outputSize == 2);

        INFO("check if input indexes are correct");
        for (auto const& name : inputNames)
        {
            REQUIRE(network->graph[name]->index == inputIndexes[name]);
        }

        INFO("check if output indexes are correct");
        for (auto const& name : outputNames)
        {
            REQUIRE(network->graph[name]->index == outputIndexes[name]);
        }
    }

    TEST_CASE("SNN.NetworkEditor.addSynapse", "[Network][NetworkEditor]") {
        std::vector<uint32_t> inputNames = { 0, 1, 2 };
        std::vector<uint32_t> outputNames = { 3, 4 };
        std::map<uint32_t, uint32_t> inputIndexes = { {0, 0}, {1, 1}, {2, 2} };
        std::map<uint32_t, uint32_t> outputIndexes = { {3, 0}, {4, 1} };

        Network* network = new Network();

        for (auto const& input : inputIndexes)
        {
            NetworkEditor::addNode(*network, input.second, Node::NodeMode::input);
        }

        for (auto const& input : outputIndexes)
        {
            NetworkEditor::addNode(*network, input.second, Node::NodeMode::output);
        }

        NetworkEditor::addSynapse(*network, *network->graph[0].get(), *network->graph[3].get(), 1);
        REQUIRE(network->graph[0]->conn[0]->dest == network->graph[3].get());
        REQUIRE(network->graph[3]->sources[0]->src == network->graph[0].get());

        for (auto const& [name, node] : network->graph) 
        {
            if (name != 0)
                REQUIRE_THAT(node->conn, Catch::Matchers::IsEmpty());
            else
                REQUIRE(node->conn.size() == 1);
            if (name != 3)
                REQUIRE_THAT(node->sources, Catch::Matchers::IsEmpty());
            else
                REQUIRE(node->sources.size() == 1);
        }

        NetworkEditor::addSynapse(*network, *network->graph[0].get(), *network->graph[4].get(), 1);
        REQUIRE(network->graph[0]->conn[1]->dest == network->graph[4].get());
        REQUIRE(network->graph[4]->sources[0]->src == network->graph[0].get());

        for (auto const& [name, node] : network->graph)
        {
            if (name != 0)
                REQUIRE_THAT(node->conn, Catch::Matchers::IsEmpty());
            else
                REQUIRE(node->conn.size() == 2);
            if (name != 3 && name != 4)
                REQUIRE_THAT(node->sources, Catch::Matchers::IsEmpty());
            else
                REQUIRE(node->sources.size() == 1);
        }
    }

    TEST_CASE("SNN.NetworkEditor.removeNode", "[Network][NetworkEditor]") {
        Network* network = new Network();
        REQUIRE_NOTHROW(NetworkLoader::load("2net.txt", *network));
        const uint32_t id = 2;
        REQUIRE(network->graph.find(id) != network->graph.end());

        NetworkEditor::removeNode(*network, *network->graph[id]);

        REQUIRE(network->graph.find(id) == network->graph.end());

        for (auto const& [name, node] : network->graph)
        {
            for (auto const& conn : node->conn)
            {
                REQUIRE(conn->dest->name != id);
            }
            for (auto const& src : node->sources)
            {
                REQUIRE(src->src->name != id);
            }
        }
    }

    TEST_CASE("SNN.NetworkEditor.removeSynapse", "[Network][NetworkEditor]") {
        Network* network = new Network();
        REQUIRE_NOTHROW(NetworkLoader::load("2net.txt", *network));

        Synapse* synapse = network->graph[1]->conn[0].get();
        Node* src = synapse->src;
        Node* dest = synapse->dest;

        NetworkEditor::removeSynapse(*network, *src, *dest);

        for (auto const& [name, node] : network->graph)
        {
            for (auto const& conn : node->conn)
            {
                REQUIRE((conn->src != src || conn->dest != dest));
            }
            for (auto const& source : node->sources)
            {
                REQUIRE((source->src != src || source->dest != dest));
            }
        }
    }
}