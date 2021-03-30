# INFO
It is my own implementation of the Spiking Neural Network system for simulation purposes. 

It is a simple version with support of a single thread CPU or GPU acceleration. 

Read txt or bin file with adjacency matrix merged with weight matrix of the network, apply input signal and simulate.

After reading input file you need to call BFSSort on the network object to slice the network and make layered structure of the network. Layered structure of the network is especially required by the GPU implementation, because it runs layer after layer simulation.

Currently there is available only Izhikevich model. I will add more models in the future

## INPUT FILE FORMAT
### TXT FILE
separator: space

first line is a list of input neuron indexes

second line is a list of output neuron indexes

next there is an adjacency matrix merged with weights

example matrix
```
0 1
5 6
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
1. 8 bits (uint8) for settings
    1. precision (1 if float, 0 if double)
    1. not used
    1. not used
    1. not used
    1. not used
    1. not used
    1. not used
    1. not used
1. 32 bits (uint32) num of all neurons N
1. 32 bits (uint32) num of input neurons I
1. 32 bits (uint32) num of output neurons O
1. input neuron indexes (len of I)
1. output neuron indexes (len of O)
1. adjacency matrix (len of N*N) with specified precision (float or double)
# TODO
- [x] working CPU single threaded version
- [x] working GPU version
- [ ] save network to txt and bin file
- [ ] multithread CPU version
- [ ] template class with precision sellection of the Neuron and Synapse
- [ ] more neuron models