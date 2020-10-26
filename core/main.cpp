#include "src/network.hpp"
#include "src/networkLoader.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

int main()
{	
	SNN::Network* network = new SNN::Network();
	SNN::NetworkLoader::load("../../testnet.txt", network);
	network->BFSSort();

	//std::fstream file;
	//file.open("out.txt", std::ios::out);
	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < 100; i++)
	{
		std::vector<double> a = network->run({ 10, 10, 10 });
		//file << a[0] << " " << a[1] << std::endl;
	}
	//file.close();
	auto end = std::chrono::steady_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	std::cout << diff.count() << std::endl;
}