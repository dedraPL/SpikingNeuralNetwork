#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "neuron.hpp"
#include "synapse.hpp"
#include "networkLoader.hpp"
#include "network.hpp"

namespace py = pybind11;

PYBIND11_MODULE(snn, m) {
	py::class_<SNN::Neuron>(m, "Neuron")
		.def(py::init<std::string, NEURON_TYPE, NEURON_TYPE, NEURON_TYPE, NEURON_TYPE, uint32_t>())
		.def("AddCurrent", &SNN::Neuron::AddCurrent)
		.def("CalculatePotential", &SNN::Neuron::CalculatePotential)
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
		//.def("run", &SNN::Network::run)
		.def("run", static_cast<std::vector<double>(SNN::Network::*)(std::vector<double>)>(&SNN::Network::run))
		.def("rund", &SNN::Network::rund)
		.def("runf", &SNN::Network::runf)
		.def_readwrite("graph", &SNN::Network::graph)
		.def_readwrite("graphOrder", &SNN::Network::graphOrder)
		;

	py::class_<SNN::Network::Node>(m, "Node")
		.def("update", &SNN::Network::Node::update)
		.def_readwrite("node", &SNN::Network::Node::node, py::return_value_policy::reference)
		.def_readwrite("conn", &SNN::Network::Node::conn)
		.def_readwrite("sources", &SNN::Network::Node::sources)
		;

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