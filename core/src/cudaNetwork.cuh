#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <vector>
#include "network.hpp"

namespace SNN {
	class CUDANetwork : public Network{
	public:
		CUDANetwork();
		~CUDANetwork();
		std::vector<float> run(std::vector<float> inputs);
		std::vector<float> runContinuous(std::vector<float> inputs, uint32_t times);
		void prepareMemory();
		void copyToGPU();
		void copytFromGPU();
		void freeGPUMemory();

		float* V;
		float* U;
	private:
		std::vector<float> retrieveOutput();
		void propagateInput();

		float* R;
		uint32_t* connections2DArray; 
		uint32_t* connectionsPointers;
		uint32_t connectionsNumber = 0;
		float** outputs;

		float** indexes;
		uint32_t* connectionsIndexesNumber;
		std::vector<std::vector<uint32_t>> W;
		std::vector<std::vector<uint32_t>> H;

		float* deviceV;
		float* deviceU;
		float* deviceR;
		uint32_t* deviceConnections2DArray;
		uint32_t* deviceConnectionsPointers;

		float* deviceInputs;
		uint32_t** deviceIndexes;
		float** deviceI;
		float** deviceIMatrix;
		uint32_t** deviceW;
		uint32_t** deviceH;

		const unsigned block_x = 32;
		const unsigned block_y = 32;
		const dim3 nTPB;
	};
}