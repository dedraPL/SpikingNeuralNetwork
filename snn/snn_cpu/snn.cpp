#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "node.hpp"
#include "neuron.hpp"
#include "synapse.hpp"
#include "networkLoader.hpp"
#include "networkEditor.hpp"
#include "network.hpp"
#include "encoders/encoder.hpp"
#include "decoders/decoder.hpp"
#include "encoders/transparent.hpp"
#include "decoders/transparent.hpp"
#include "decoders/averageOverTime.hpp"
#include "decoders/binary.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

template<template<typename> class T, typename type>
void declareEncoder(py::module& m, const char* className)
{
	py::class_<T<type>, SNN::Encoder::IEncoder<type>>(m, className)
		.def(py::init<>())
		.def("encode", &T<type>::encode);
}

template<template<typename> class T, typename type>
void declareDecoder(py::module& m, const char* className)
{
	py::class_<T<type>, SNN::Decoder::IDecoder<type>>(m, className)
		.def(py::init<>())
		.def("decode", &T<type>::decode);
}

template<typename type>
void declareEncoders(py::module& m, const char* typestr)
{
	using vector = Eigen::Matrix<type, Eigen::Dynamic, 1>;
	class PyAbstract : public SNN::Encoder::IEncoder<type> {
	public:
		using SNN::Encoder::IEncoder<type>::IEncoder;
		vector encode(const Eigen::Ref<const vector>& input) override {
			PYBIND11_OVERRIDE_PURE(
				vector,
				SNN::Encoder::IEncoder<type>,
				encode,
				input
			);
		}
	};

	std::string pyclass_name = std::string("IEncoder") + std::string(typestr);
	py::class_<SNN::Encoder::IEncoder<type>, PyAbstract>(m, pyclass_name.c_str())
		.def("encode", &SNN::Encoder::IEncoder<type>::encode);

	pyclass_name = std::string("Transparent") + std::string(typestr);
	declareEncoder<SNN::Encoder::Transparent, type>(m, pyclass_name.c_str());
}

template<typename type>
void declareDecoders(py::module& m, const char* typestr)
{
	using vector = Eigen::Matrix<type, Eigen::Dynamic, 1>;
	class PyAbstract : public SNN::Decoder::IDecoder<type> {
	public:
		using SNN::Decoder::IDecoder<type>::IDecoder;
		vector decode(const Eigen::Ref<const vector>& input) override {
			PYBIND11_OVERRIDE_PURE(
				vector,
				SNN::Decoder::IDecoder<type>,
				decode,
				input
			);
		}
	};

	std::string pyclass_name = std::string("IDecoder") + std::string(typestr);
	py::class_<SNN::Decoder::IDecoder<type>, PyAbstract>(m, pyclass_name.c_str())
		.def("decode", &SNN::Decoder::IDecoder<type>::decode);

	pyclass_name = std::string("Transparent") + std::string(typestr);
	py::class_<SNN::Decoder::Transparent<type>, SNN::Decoder::IDecoder<type>>(m, pyclass_name.c_str())
		.def(py::init<>())
		.def("decode", &SNN::Decoder::Transparent<type>::decode);

	pyclass_name = std::string("AverageOverTime") + std::string(typestr);
	py::class_<SNN::Decoder::AverageOverTime<type>, SNN::Decoder::IDecoder<type>>(m, pyclass_name.c_str())
		.def(py::init<uint32_t, uint32_t>())
		.def("decode", &SNN::Decoder::AverageOverTime<type>::decode);

	pyclass_name = std::string("Binary") + std::string(typestr);
	py::class_<SNN::Decoder::Binary<type>, SNN::Decoder::IDecoder<type>>(m, pyclass_name.c_str())
		.def(py::init<>())
		.def("decode", &SNN::Decoder::Binary<type>::decode);
}

PYBIND11_MODULE(snn, m) {
	m.doc() = "Spiking Neural Network module";
	py::module encoders_module = m.def_submodule("Encoders");
	py::module decoders_module = m.def_submodule("Decoders");

	auto Node = py::class_<SNN::Node, std::shared_ptr<SNN::Node>>(m, "Node", "Node of the network graph")
		.def(py::init<uint32_t, uint32_t>(), "name"_a, "index"_a=0)
		.def_readwrite("node", &SNN::Node::node, py::return_value_policy::reference, "returns Neuron of the Node")
		.def_readwrite("conn", &SNN::Node::conn, "returns vector of outcoming synapses")
		.def_readwrite("sources", &SNN::Node::sources, "returns vector of incoming synapses")
		.def_readonly("mode", &SNN::Node::mode, "mode of the Node")
		.def_readwrite("name", &SNN::Node::name, "name of the Node")
		.def_readwrite("index", &SNN::Node::index, "index of the Node")
		;

	py::enum_<SNN::Node::NodeMode>(Node, "NodeMode", "Mode of the Node")
		.value("input", SNN::Node::NodeMode::input)
		.value("hidden", SNN::Node::NodeMode::hidden)
		.value("output", SNN::Node::NodeMode::output)
		.export_values()
		;

	py::class_<SNN::Neuron, std::shared_ptr<SNN::Neuron>>(m, "Neuron", "Neuron class. Part of the Node.")
		.def(py::init<NEURON_TYPE, NEURON_TYPE, NEURON_TYPE, NEURON_TYPE>(), "a"_a, "b"_a, "c"_a, "d"_a, "a, b, c, d are Izhikevich model parameters")
		.def("AddCurrent", &SNN::Neuron::AddCurrent, "current"_a, "add current to the Neuron accumulator. Current is flushed when calling CalculatePotential method and set back to 0")
		.def("CalculatePotential", &SNN::Neuron::CalculatePotential, "Calculaters potential of the membrane and sets current accumulator back to 0. Retruns V and U voltages respectively")
		.def_readwrite("a", &SNN::Neuron::a)
		.def_readwrite("b", &SNN::Neuron::b)
		.def_readwrite("c", &SNN::Neuron::c)
		.def_readwrite("d", &SNN::Neuron::d)
		.def_readwrite("prevV", &SNN::Neuron::prevV, "previous V voltage")
		.def_readwrite("prevU", &SNN::Neuron::prevU, "previous U voltage")
		.def_readwrite("current", &SNN::Neuron::current, "current current of the accumulator")
		;

	py::class_<SNN::Synapse, std::shared_ptr<SNN::Synapse>>(m, "Synapse", "Synapse and graph's edge class")
		.def(py::init<SNN::Node*, SNN::Node*, SYNAPSE_TYPE>(), "src"_a, "dest"_a, "r"_a, "New synapse between src Node and dest Node with r resistance")
		.def("CalculateCurrent", &SNN::Synapse::CalculateCurrent, "v1"_a, "v2"_a, "calculates current between v1 and v2 voltages with r resistor")
		.def("ChangeResistance", &SNN::Synapse::ChangeResistance, "r"_a, "change Synapse resistance")
		.def_readwrite("r", &SNN::Synapse::r, "synapse resistance")
		.def_readwrite("dest", &SNN::Synapse::dest, "destination Node")
		.def_readwrite("src", &SNN::Synapse::src, "source Node")
		;

	py::class_<SNN::NetworkLoader>(m, "NetworkLoader", "Network loading and saving tools")
		.def_static("load", &SNN::NetworkLoader::load, "filename"_a, "network"_a, "load network architecture from filename into network")
		.def_static("loadBin", &SNN::NetworkLoader::loadBin, "filename"_a, "network"_a, "load network architecture from binary filename into network")
		.def_static("save", &SNN::NetworkLoader::save, "filename"_a, "network"_a, "save network architecture from network into filename")
		.def_static("saveBin", &SNN::NetworkLoader::saveBin, "filename"_a, "network"_a, "save binary network architecture from network into filename")
		;

	py::class_<SNN::NetworkEditor>(m, "NetworkEditor", "Network editing tools")
		.def_static("addHiddenNode", &SNN::NetworkEditor::addHiddenNode, "network"_a, "add new hidden Node with next possible name and return it")
		.def_static("addNode", &SNN::NetworkEditor::addNode, "network"_a, "index"_a, "mode"_a, "add new Node with name and mode and return it")
		.def_static("addSynapse", &SNN::NetworkEditor::addSynapse, "network"_a, "source"_a, "destination"_a, "r"_a, "add new Synapse between source and destination Nodes with r resistance")
		.def_static("removeNode", &SNN::NetworkEditor::removeNode, "network"_a, "node"_a, "remove node Node from network and all incoming and outcoming Synapses")
		.def_static("removeSynapse", &SNN::NetworkEditor::removeSynapse, "network"_a, "source"_a, "destination"_a "remove Synapse between source and destination Nodes")
		;

	py::class_<SNN::Network>(m, "Network", "Network class")
		.def(py::init<>())
		.def("BFSSort", &SNN::Network::BFSSort, "sort graph into layers that can be ran sequentially. Returns vector of Node's names in corresponding layers")
		//.def("run", &SNN::Network::run)
		.def("run", static_cast<std::vector<double>(SNN::Network::*)(std::vector<double>)>(&SNN::Network::run), "inputs"_a, "apply inputs vector and propagate signals through the Network. DEPRECATED")
		.def("rund", &SNN::Network::rund, "inputs"_a, "apply inputs vector and propagate signals through the Network. Float precision")
		.def("runf", &SNN::Network::runf, "inputs"_a, "apply inputs vector and propagate signals through the Network. Double precision")
		.def("getGraph", &SNN::Network::getGraph, py::return_value_policy::reference, "returns graph map")
		.def("getInputNodes", &SNN::Network::getInputsIdx, py::return_value_policy::reference, "returns vector of inputs indexes")
		.def("getOutputNodes", &SNN::Network::getOutputsIdx, py::return_value_policy::reference, "returns vector of outputs indexes")
		.def_readwrite("graph", &SNN::Network::graph, py::return_value_policy::reference, "map of graph")
		.def_readwrite("graphOrder", &SNN::Network::graphOrder, "vector of Node's names in corresponding layers")
		;

	declareEncoders<double>(encoders_module, "d");
	declareEncoders<float>(encoders_module, "f");

	declareDecoders<double>(decoders_module, "d");
	declareDecoders<float>(decoders_module, "f");

	static py::exception<SNN::NetworkLoader::FileNotFoundError> exFileNotFoundError(m, "FileNotFoundError");
	py::register_exception_translator([](std::exception_ptr p) {
		try {
			if (p) std::rethrow_exception(p);
		}
		catch (const SNN::NetworkLoader::FileNotFoundError& e) {
			exFileNotFoundError(e.what());
		}
		});

	static py::exception<SNN::NetworkLoader::InvalidNetworkError> exInvalidNetworkError(m, "InvalidNetworkError");
	py::register_exception_translator([](std::exception_ptr p) {
		try {
			if (p) std::rethrow_exception(p);
		}
		catch (const SNN::NetworkLoader::InvalidNetworkError& e) {
			exInvalidNetworkError(e.what());
		}
		});

	static py::exception<SNN::Network::InputSizeError> exInputSizeError(m, "InputSizeError");
	py::register_exception_translator([](std::exception_ptr p) {
		try {
			if (p) std::rethrow_exception(p);
		}
		catch (const SNN::Network::InputSizeError& e) {
			exInputSizeError(e.what());
		}
		});
}