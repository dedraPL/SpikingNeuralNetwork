#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "neuron.hpp"
#include "synapse.hpp"
#include "networkLoader.hpp"
#include "network.hpp"
#include "encoders/encoder.hpp"
#include "decoders/decoder.hpp"
#include "encoders/transparent.hpp"
#include "decoders/transparent.hpp"
#include "decoders/averageOverTime.hpp"
#include "decoders/binary.hpp"

namespace py = pybind11;

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
	py::module encoders_module = m.def_submodule("Encoders");
	py::module decoders_module = m.def_submodule("Decoders");

	py::class_<SNN::Neuron>(m, "Neuron")
		.def(py::init<std::string, NEURON_TYPE, NEURON_TYPE, NEURON_TYPE, NEURON_TYPE, uint32_t>())
		.def("AddCurrent", &SNN::Neuron::AddCurrent)
		.def("CalculatePotential", &SNN::Neuron::CalculatePotential)
		.def_readwrite("name", &SNN::Neuron::name)
		.def_readwrite("a", &SNN::Neuron::a)
		.def_readwrite("b", &SNN::Neuron::b)
		.def_readwrite("c", &SNN::Neuron::c)
		.def_readwrite("d", &SNN::Neuron::d)
		.def_readwrite("prevV", &SNN::Neuron::prevV)
		.def_readwrite("prevU", &SNN::Neuron::prevU)
		.def_readwrite("current", &SNN::Neuron::current)
		;

	py::class_<SNN::Synapse>(m, "Synapse")
		.def(py::init<uint32_t, SYNAPSE_TYPE>())
		.def("CalculateCurrent", &SNN::Synapse::CalculateCurrent)
		.def("ChangeResistance", &SNN::Synapse::ChangeResistance)
		.def_readwrite("r", &SNN::Synapse::r)
		.def_readwrite("dest", &SNN::Synapse::dest)
		;

	py::class_<SNN::NetworkLoader>(m, "NetworkLoader")
		.def_static("load", &SNN::NetworkLoader::load)
		.def_static("loadBin", &SNN::NetworkLoader::loadBin)
		.def_static("save", &SNN::NetworkLoader::save)
		.def_static("saveBin", &SNN::NetworkLoader::saveBin)
		;

	py::class_<SNN::Network>(m, "Network")
		.def(py::init<>())
		.def("BFSSort", &SNN::Network::BFSSort)
		.def("run", &SNN::Network::run)
		.def_readwrite("graph", &SNN::Network::graph)
		.def_readwrite("graphOrder", &SNN::Network::graphOrder)
	py::class_<SNN::Network> Network(m, "Network");
	Network.def(py::init<>());
	Network.def("BFSSort", &SNN::Network::BFSSort);
	//.def("run", &SNN::Network::run)
	Network.def("run", static_cast<std::vector<double>(SNN::Network::*)(std::vector<double>)>(&SNN::Network::run));
	Network.def("rund", &SNN::Network::rund);
	Network.def("runf", &SNN::Network::runf);
	Network.def("getGraph", &SNN::Network::getGraph, py::return_value_policy::reference);
	Network.def_readwrite("graph", &SNN::Network::graph, py::return_value_policy::reference);
	Network.def_readwrite("graphOrder", &SNN::Network::graphOrder);

		;

	py::class_<SNN::Network::Node>(m, "Node")
		.def(py::init<>())
		.def("update", &SNN::Network::Node::update)
		.def_readwrite("node", &SNN::Network::Node::node, py::return_value_policy::reference)
		.def_readwrite("conn", &SNN::Network::Node::conn)
		.def_readwrite("sources", &SNN::Network::Node::sources)
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
			// Set MyException as the active python error
			exFileNotFoundError(e.what());
		}
		});

	static py::exception<SNN::NetworkLoader::InvalidNetworkError> exInvalidNetworkError(m, "InvalidNetworkError");
	py::register_exception_translator([](std::exception_ptr p) {
		try {
			if (p) std::rethrow_exception(p);
		}
		catch (const SNN::NetworkLoader::InvalidNetworkError& e) {
			// Set MyException as the active python error
			exInvalidNetworkError(e.what());
		}
		});

	static py::exception<SNN::Network::InputSizeError> exInputSizeError(m, "InputSizeError");
	py::register_exception_translator([](std::exception_ptr p) {
		try {
			if (p) std::rethrow_exception(p);
		}
		catch (const SNN::Network::InputSizeError& e) {
			// Set MyException as the active python error
			exInputSizeError(e.what());
		}
		});
}