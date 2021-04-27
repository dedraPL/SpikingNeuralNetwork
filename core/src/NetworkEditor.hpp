#pragma once

#include "neuron.hpp"
#include "network.hpp"

namespace SNN {
	class NetworkEditor {
	public:
		static std::shared_ptr<Neuron> addHiddenNode(Network& network);

		static std::shared_ptr<Neuron> addNode(Network& network, uint32_t index, Network::NodeMode mode);

		static std::shared_ptr<Synapse> addSynapse(Network& network, Neuron& source, Neuron& destination, SYNAPSE_TYPE r);

		static void removeNode(Network& network, Network::Node& node);

		static void removeSynapse(Network& network, Neuron& source, Neuron& destination);
	};
}

