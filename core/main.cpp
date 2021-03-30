#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "src/network.hpp"
#include "src/cudaNetwork.cuh"
#include "src/networkLoader.hpp"
#include <iostream>
#include <locale>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>

template <class charT, charT sep>
class punct_facet : public std::numpunct<charT> {
protected:
	charT do_decimal_point() const { return sep; }
};

int main()
{
	uint32_t num_of_calls = 100;
	uint32_t gpuTime, cpuTime;
	bool enableDumpToFile = true;
	uint8_t floatPrintPrecision = 17;
	std::string netFileFilename = "C:\\Users\\drozd\\Documents\\Python Scripts\\snn\\testnet3.txt";
	{
		SNN::CUDANetwork* network = new SNN::CUDANetwork();
		SNN::NetworkLoader::loadBin(netFileFilename, (SNN::Network*)network);
		network->BFSSort();
		network->prepareMemory();
		network->copyToGPU();

		auto start = std::chrono::high_resolution_clock::now();
		network->runContinuous({ 10, 10, 10 }, num_of_calls);
		cudaDeviceSynchronize();
		auto end = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		std::cout << "gpu time " << diff.count() << " ns" << std::endl;
		gpuTime = diff.count();

		if (enableDumpToFile)
		{
			std::fstream dumpFile;
			dumpFile.open("gpuDump.txt", std::ios::out);
			dumpFile.imbue(std::locale(std::cout.getloc(), new punct_facet<char, ','>));
			dumpFile << std::setprecision(floatPrintPrecision);
			for (int i = 0; i < network->graph.size(); i++)
			{
				dumpFile << network->V[i] << " " << network->U[i] << std::endl;
			}
			dumpFile.close();
		}
	}
	{
		SNN::Network* network = new SNN::Network();
		SNN::NetworkLoader::loadBin(netFileFilename, network);
		network->BFSSort();

		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < num_of_calls; i++)
		{
			std::vector<double> a = network->run({ 10, 10, 10 });
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		std::cout << "cpu time " << diff.count() << " ns" << std::endl;
		cpuTime = diff.count();

		if (enableDumpToFile)
		{
			std::fstream dumpFile;
			dumpFile.open("cpuDump.txt", std::ios::out);
			dumpFile.imbue(std::locale(std::cout.getloc(), new punct_facet<char, ','>));
			dumpFile << std::setprecision(floatPrintPrecision);
			for (auto const& el : network->graph)
			{
				dumpFile << el.second->node->prevV << " " << el.second->node->prevU << std::endl;
			}
			dumpFile.close();
		}
	}
	std::cout << "cpu/gpu: " << cpuTime / (double)gpuTime << std::endl;
	std::cout << "gpu/cpu: " << gpuTime / (double)cpuTime << std::endl;
}