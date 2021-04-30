#pragma once

#include "neuron.hpp"
#include "network.hpp"
#include "node.hpp"

namespace SNN {
	class NetworkEditor {
	public:
		static std::shared_ptr<Node> addHiddenNode(Network& network);

		static std::shared_ptr<Node> addNode(Network& network, uint32_t index, Node::NodeMode mode);

		static std::shared_ptr<Synapse> addSynapse(Network& network, Node& source, Node& destination, SYNAPSE_TYPE r);

		static void removeNode(Network& network, Node& node);

		static void removeSynapse(Network& network, Node& source, Node& destination);
	};
}

