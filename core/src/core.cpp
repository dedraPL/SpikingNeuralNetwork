#pragma once

#include <boost/python.hpp>
#include <boost/python/exception_translator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <map>

#include "synapse.hpp"
#include "neuron.hpp"
#include "network.hpp"
#include "networkLoader.hpp"

template <class K, class V, bool NoProxy = true>
struct map_to_dict {
    static PyObject* convert(const std::map<K, V>& map) {
        boost::python::dict* dict = new boost::python::dict();
        typename std::map<K, V>::const_iterator it;
        for (it = map.begin(); it != map.end(); ++it) {
            if (NoProxy) {
                dict->setdefault(it->first, boost::ref(it->second));
            } else {
                dict->setdefault(it->first, it->second);
            }
        }
        return dict->ptr();
    }
    static PyTypeObject const* get_pytype() { return &PyDict_Type; }
};

void translate(SNN::NetworkLoader::FileNotFoundError const& e)
{
    PyErr_SetString(PyExc_FileNotFoundError, e.what());
}

BOOST_PYTHON_MODULE(core) {
    using namespace boost::python;

    register_exception_translator<SNN::NetworkLoader::FileNotFoundError>(&translate);

    class_<SNN::Synapse>("Synapse", init<uint32_t, float>())
        .def("CalculateCurrent", &SNN::Synapse::CalculateCurrent)
        .def("ChangeResistance", &SNN::Synapse::ChangeResistance)
        .def_readonly("r", &SNN::Synapse::r)
        .def_readonly("dest", &SNN::Synapse::dest)
    ;

    class_<SNN::Neuron>("Neuron", init<std::string, float, float, float, float, optional<unsigned int>>())
        .def("CalculatePotential", &SNN::Neuron::CalculatePotential)
        .def_readwrite("name", &SNN::Neuron::name)
        .def_readwrite("a", &SNN::Neuron::a)
        .def_readwrite("b", &SNN::Neuron::b)
        .def_readwrite("c", &SNN::Neuron::c)
        .def_readwrite("d", &SNN::Neuron::d)
        .def_readwrite("index", &SNN::Neuron::index)
    ;

    class_<SNN::Network>("Network", init<>())
        .add_property("graph", &SNN::Network::getGraph)
    ;

    enum_<SNN::Network::NodeMode>("NodeMode")
        .value("input", SNN::Network::NodeMode::input)
        .value("output", SNN::Network::NodeMode::output)
        .value("hindden", SNN::Network::NodeMode::hidden)
    ;

    class_<SNN::Network::Node>("Node", init<>())
        //.def_readwrite("node", &SNN::Network::Node::node)
        //.def_readwrite("mode", &SNN::Network::Node::mode)
        //.def_readwrite("conn", &SNN::Network::Node::conn)
        //.def_readwrite("sources", &SNN::Network::Node::sources)
        .add_property("node", &SNN::Network::Node::getNode)
        .add_property("mode", &SNN::Network::Node::getMode)
        .add_property("conn", &SNN::Network::Node::getConn)
        .add_property("sources", &SNN::Network::Node::getSources)
        //.def("getNode", &SNN::Network::Node::getNode, return_value_policy<reference_existing_object>())
    ;

    class_<SNN::NetworkLoader>("NetworkLoader", no_init)
        .def("load", &SNN::NetworkLoader::load)
    ;

    class_<std::map<uint32_t, SNN::Network::Node*>>("mapuint32Node")
        .def(map_indexing_suite<std::map<uint32_t, SNN::Network::Node*>, true>())
    ;

    //to_python_converter<std::map<uint32_t, SNN::Network::Node*>, map_to_dict<uint32_t, SNN::Network::Node*>, true>();
}