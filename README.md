# INFO
It is my own implementation of the Spiking Neural Network system for simulation purposes. 

It is a simple version with support of a single thread CPU or GPU acceleration. 

Read txt or bin file with adjacency matrix merged with weight matrix of the network, apply input signal and simulate.

After reading input file you need to call BFSSort on the network object to slice the network and make layered structure of the network. Layered structure of the network is especially required by the GPU implementation, because it runs layer after layer simulation.

Currently there is available only Izhikevich model. I will add more models in the future.

Synapse is modeled as a parallel RC element

![formula](/doc/images/synapse_eq.gif)

## GPU VERSION IS CURRENTLY OUTDATED

## INPUT FILE FORMAT
Both TXT and binary format are designed with same principals. They are handled like JSON files (or MessagePack).

Mandatory fields:
* inputs
* outputs
* matrix

Optional fields:
* format
* description
* synapse_c
* network_size
* models

Some of optional fields are mandatory in binary format.

Inputs is an array of the input nodes in the network. Outputs is an array of the output nodes in the network. Matrix is an adjacency matrix merged with weights. Format of the matrix is different in TXT and binary format and described in corresponding section.

Format is a string representing precision of real number data ("f" - 32 bits, "d" - 64 bits). Description is a text description of the network. Synapse_c is a C<sub>in</sub> capacitance of every synapse in the network. If synapse_c is omitted then is setted to 0. Network_size is an count of all neurons in the network. 

Models is an array of neuron model parameters and array of node membership. Models field require models array of object of 4 parameters: a, b, c, d and nodes array of membership of every neuron in the network. First model (index 0) is default and should not be present in models array. Parameters of the default model are a = 0.02, b = 0.2, c = -65, d = 8. If whole models array is omitted, then every neuron take default parameters.

In this example, only 4th and 6th neurons have non-default parameters.
```
"models": {
    "models": [
        {
            "a": -0.02,
            "b": -1,
            "c": -65,
            "d": 8
        }
    ],
    "nodes": [
        0,
        0,
        0,
        1,
        0,
        1
    ]
}
```

### TXT FILE
All data are formated as a JSON. In TXT format, mandatory fields: inputs, outputs and matrix. Matrix is an array of strings. Each element of array is signle row of array. I chose this format, because TXT file is intended to be easy to read by human.

example matrix
```
0 0 1 1 1 0 0
0 0 1 1 1 0 0
0 0 0 0 0 1 1
0 0 0 0 0 1 1
0 0 0 0 0 1 1
0 0 0 0 0 0 0
0 0 0 0 0 0 0
```
produces network like this

![graph](/doc/images/graph.PNG)

where all edges have weight = 1

### BIN FILE
Binary file is formated as a [MessagePack](https://msgpack.org). Mandatory fields: format, inputs, outputs, matrix, network_size. Every field is converted to the binary according to MessagePack format except matrix, with is formated as an ext8, ext16 or ext32 field. Matrix is formated as a series of float or double data and should be size of (network_size * network_size) * bytes_per_precision bytes. Format and network_size are mandatory to determine how many bytes per connection and per row loader need to read.

# KNOWN ISSUES

* loading txt file with loadBin results with huge memory consumption
* inputs and outputs index consistency cannot be guaranteed after removing input or output Node

# TODO
- [x] working CPU single threaded version
- [x] working GPU version
- [x] save network to txt and bin file
- [ ] multithread CPU version
- [ ] template class with precision sellection of the Neuron and Synapse
- [ ] more neuron models
- [ ] documentation and comments 