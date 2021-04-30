#include "cudaNetwork.cuh"

#include "utils.hpp"

namespace SNN 
{
	__device__ void izhikevich(float i, float& prevV, float& prevU, float a = 0.02, float dt = 0.5, float b = 0.2, float c = -65.0, float d = 8.0)
	{
        float Vans, Uans;
        if (prevV < 35)
        {
            float dv = (0.04 * prevV + 5) * prevV + 140 - prevU;
            Vans = prevV + (dv + i) * dt;
            float du = a * (b * prevV - prevU);
            Uans = prevU + dt * du;
            Vans = fminf(35, Vans);
            /*if (Vans > 35)
                Vans = 35;*/
        }
        else
        {
            Vans = c;
            Uans = prevU + d;
        }
        prevU = Uans;
        prevV = Vans;
	}

    __global__ void izhikevichLayerForward(float* V, float* U, const float* __restrict__ I, const float* __restrict__ R, const uint32_t* __restrict__ connections2DArray, const uint32_t* __restrict__ connectionsPointers, const uint32_t* __restrict__ indexes, uint32_t neuronNumber)
    {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < neuronNumber)
        {
            izhikevich(I[index], V[indexes[index]], U[indexes[index]]);
        }
    }

    __global__ void izhikevichInputLayerForward(const float* __restrict__ inputs, float* V, float* U, const uint32_t* __restrict__ indexes, uint32_t neuronNumber)
    {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < neuronNumber)
        {
            izhikevich(inputs[indexes[index]], V[indexes[index]], U[indexes[index]]);
        }
    }

    __global__ void synapse(const float* __restrict__ V, const float* __restrict__ R,
        const uint32_t* __restrict__ connections2DArray, const uint32_t* __restrict__ connectionsPointers, const uint32_t* __restrict__ indexes,
        const uint32_t* __restrict__ connectionsWIndexes, const uint32_t* __restrict__ connectionsHIndexes, uint32_t connectionsNumber,
        float* in, size_t width)
    {
        uint32_t index = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
        if (index < connectionsNumber)
        {
            float v_source = V[connections2DArray[connectionsPointers[indexes[connectionsWIndexes[index]]] + connectionsHIndexes[index]]];
            float v_target = V[indexes[connectionsWIndexes[index]]];
            float i = fmaxf(0, (v_source - v_target) / R[connections2DArray[connectionsPointers[indexes[connectionsWIndexes[index]]] + connectionsHIndexes[index]]]);
            in[width * connectionsHIndexes[index] + connectionsWIndexes[index]] = i;
            /*printf("index: %u in: %u target: %u source: %u\n", index, width * connectionsHIndexes[index] + connectionsWIndexes[index],
                indexes[connectionsWIndexes[index]], connections2DArray[connectionsPointers[indexes[connectionsWIndexes[index]]] + connectionsHIndexes[index]]);*/
        }
    }

    __global__ void synapse_sum(const float* __restrict__ in, float* __restrict__ out, size_t width, size_t height)
    {
        __shared__ float sdata[32][32];
        for (uint32_t w = threadIdx.x + blockDim.x * blockIdx.x; w < (width & (~((unsigned long long)(32 - 1)))) + ((width & (32 - 1)) ? 32 : 0); w += gridDim.x * blockDim.x) {          // grid-stride loop across matrix width
            sdata[threadIdx.y][threadIdx.x] = 0;
            uint32_t in_ptr = w + threadIdx.y * width;
            for (uint32_t h = threadIdx.y; h < height; h += 32) { // block-stride loop across matrix height
                sdata[threadIdx.y][threadIdx.x] += (w < width) ? in[in_ptr] : 0;
                in_ptr += width * 32;
            }
            __syncthreads();
            float my_val = sdata[threadIdx.x][threadIdx.y];
            for (int i = warpSize >> 1; i > 0; i >>= 1)                       // warp-wise parallel sum reduction
                my_val += __shfl_xor_sync(0xFFFFFFFFU, my_val, i);
            __syncthreads();
            if (threadIdx.x == 0) sdata[0][threadIdx.y] = my_val;
            __syncthreads();
            if ((threadIdx.y == 0) && ((w) < width)) out[w] = sdata[0][threadIdx.x];
        }
    }

    CUDANetwork::CUDANetwork() : nTPB{ 32, 32 }
	{
        this->V = nullptr;
        this->U = nullptr;
        this->R = nullptr;
        this->connections2DArray = nullptr;
        this->connectionsPointers = nullptr;
        this->deviceV = nullptr;
        this->deviceU = nullptr;
        this->deviceR = nullptr;
        this->deviceConnections2DArray = nullptr;
        this->deviceConnectionsPointers = nullptr;
	}

    CUDANetwork::~CUDANetwork()
    {
        freeGPUMemory();
    }

    std::vector<float> CUDANetwork::run(std::vector<float> inputs)
    {
        if (inputs.size() != inputSize)
            throw Network::InputSizeError(inputSize);

        std::vector<float> in(inputs.begin(), inputs.end());
        cudaMemcpyAsync(this->deviceInputs, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
        propagateInput();

        copytFromGPU();
        return retrieveOutput();
    }

    std::vector<float> CUDANetwork::runContinuous(std::vector<float> inputs, uint32_t times)
    {
        if (inputs.size() != inputSize)
            throw Network::InputSizeError(inputSize);

        std::vector<float> in(inputs.begin(), inputs.end());
        cudaMemcpyAsync(this->deviceInputs, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
        for (uint32_t i = 0; i < times; i++)
        {
            propagateInput();
        }
        
        copytFromGPU();
        return retrieveOutput();
    }

    void CUDANetwork::prepareMemory()
    {
        std::vector<float> tmpV;
        std::vector<float> tmpU;
        std::vector<float> tmpR;
        std::vector<uint32_t> tmpConnections2DArray;
        std::vector<uint32_t> tmpConnectionsPointers;
        this->outputs = new float* [this->outputSize];
        std::vector<uint32_t> outputIndexes(this->outputSize);

        for (auto const& node : this->graph)
        {
            tmpV.push_back(node.second->node->prevV);
            tmpU.push_back(node.second->node->prevU);
            tmpConnectionsPointers.push_back(tmpConnections2DArray.size());
            for (auto const& source : node.second->sources)
            {
                auto tmp = std::find_if(graph[source->dest->name]->conn.begin(), graph[source->dest->name]->conn.end(), [&](const std::shared_ptr<Synapse> s) { return s->dest->index == node.first; });
                tmpR.push_back((*tmp)->r);
                tmpConnections2DArray.push_back(source->dest->name);
            }
            if (node.second->mode == Node::NodeMode::output)
            {
                outputIndexes[node.second->index] = tmpV.size() - 1;
            }
        }

        this->connectionsIndexesNumber = new uint32_t[this->graphOrder.size() - 1]{ 0 };
        for (uint32_t i = 1; i < this->graphOrder.size(); ++i)
        {
            std::vector<uint32_t> w;
            std::vector<uint32_t> h;
            uint32_t wIndex = 0;
            for (auto const& neuronIndex : this->graphOrder[i])
            {
                uint32_t hIndex = 0;
                for (auto const& synapseIndex : graph[neuronIndex]->sources)
                {
                    w.push_back(wIndex);
                    h.push_back(hIndex);
                    connectionsIndexesNumber[i - 1]++;
                    hIndex++;
                }
                wIndex++;
            }
            this->W.push_back(w);
            this->H.push_back(h);
        }

        tmpConnectionsPointers.push_back(tmpConnections2DArray.size());
        this->V = new float[tmpV.size()];
        std::copy(tmpV.begin(), tmpV.end(), this->V);
        this->U = new float[tmpU.size()];
        std::copy(tmpU.begin(), tmpU.end(), this->U);
        this->R = new float[tmpR.size()];
        std::copy(tmpR.begin(), tmpR.end(), this->R);
        this->connections2DArray = new uint32_t[tmpConnections2DArray.size()];
        std::copy(tmpConnections2DArray.begin(), tmpConnections2DArray.end(), this->connections2DArray);
        this->connectionsPointers = new uint32_t[tmpConnectionsPointers.size()];
        std::copy(tmpConnectionsPointers.begin(), tmpConnectionsPointers.end(), this->connectionsPointers);
        this->connectionsNumber = tmpConnections2DArray.size(); 
        
        {
            uint32_t i = 0;
            for (auto const& index : outputIndexes)
            {
                this->outputs[i] = &V[index];
                i++;
            }
        }

        if (this->deviceV != nullptr)
        {
            freeGPUMemory();
        }

        cudaMalloc(&this->deviceV, this->graph.size() * sizeof(float));
        cudaMalloc(&this->deviceU, this->graph.size() * sizeof(float));
        cudaMalloc(&this->deviceR, this->connectionsNumber * sizeof(float));
        cudaMalloc(&this->deviceConnections2DArray, this->connectionsNumber * sizeof(float));
        cudaMalloc(&this->deviceConnectionsPointers, (this->graph.size() + 1) * sizeof(uint32_t));

        cudaMalloc(&this->deviceInputs, this->graphOrder[0].size() * sizeof(float));
        this->deviceIndexes = new uint32_t* [this->graphOrder.size()];
        this->deviceI = new float* [this->graphOrder.size() - 1];
        this->deviceIMatrix = new float* [this->graphOrder.size() - 1];
        this->deviceW = new uint32_t* [this->graphOrder.size() - 1];
        this->deviceH = new uint32_t* [this->graphOrder.size() - 1];
        for (uint32_t i = 0; i < this->graphOrder.size(); i++)
        {
            cudaMalloc(&(this->deviceIndexes[i]), this->graphOrder[i].size() * sizeof(uint32_t));
            if (i > 0)
            {
                cudaMalloc(&(this->deviceI[i - 1]), this->graphOrder[i].size() * sizeof(float));
                cudaMalloc(&(this->deviceIMatrix[i - 1]), this->graphOrder[i - 1].size() * this->graphOrder[i].size() * sizeof(float));
                cudaMalloc(&(this->deviceW[i - 1]), this->W[i - 1].size() * sizeof(uint32_t));
                cudaMalloc(&(this->deviceH[i - 1]), this->H[i - 1].size() * sizeof(uint32_t));
            }
        }
        gpuErrchk();
    }

    void CUDANetwork::copyToGPU()
    {
        cudaMemcpyAsync(this->deviceV, this->V, this->graph.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(this->deviceU, this->U, this->graph.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(this->deviceR, this->R, this->connectionsNumber * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(this->deviceConnections2DArray, this->connections2DArray, this->connectionsNumber * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(this->deviceConnectionsPointers, this->connectionsPointers, (this->graph.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);

        for (uint32_t i = 0; i < this->graphOrder.size(); i++)
        {
            cudaMemcpyAsync(this->deviceIndexes[i], this->graphOrder[i].data(), this->graphOrder[i].size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            if (i > 0)
            {
                cudaMemsetAsync(this->deviceIMatrix[i - 1], 0, this->graphOrder[i - 1].size() * this->graphOrder[i].size() * sizeof(float));
                cudaMemcpyAsync(this->deviceW[i - 1], this->W[i - 1].data(), this->W[i - 1].size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(this->deviceH[i - 1], this->H[i - 1].data(), this->H[i - 1].size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            }
        }

        cudaThreadSynchronize();
        gpuErrchk();
    }

    void CUDANetwork::copytFromGPU()
    {
        cudaMemcpyAsync(this->V, this->deviceV, this->graph.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(this->U, this->deviceU, this->graph.size() * sizeof(float), cudaMemcpyDeviceToHost);
        //cudaMemcpy(this->R, this->deviceR, this->connectionsNumber * sizeof(float), cudaMemcpyDeviceToHost);
        //cudaMemcpy(this->connections2DArray, this->deviceConnections2DArray, this->connectionsNumber * sizeof(float), cudaMemcpyDeviceToHost);
        //cudaMemcpy(this->connectionsPointers, this->deviceConnectionsPointers, (this->graph.size() + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //cudaThreadSynchronize();
        gpuErrchk();
    }

    void CUDANetwork::freeGPUMemory()
    {
        cudaFree(&this->deviceV);
        cudaFree(&this->deviceU);
        cudaFree(&this->deviceR);
        cudaFree(&this->deviceConnections2DArray);
        cudaFree(&this->deviceConnectionsPointers);

        cudaFree(&this->deviceInputs);
        for (uint32_t i = 0; i < this->graph.size(); i++)
        {
            cudaFree(&this->deviceIndexes[i]);
            if (i > 0)
            {
                cudaFree(&this->deviceI[i - 1]);
                cudaFree(&this->deviceIMatrix[i - 1]);
                cudaFree(&this->deviceW[i - 1]);
                cudaFree(&this->deviceH[i - 1]);
            }
        }
        cudaFree(&this->deviceIndexes);
        cudaFree(&this->deviceI);
        cudaFree(&this->deviceIMatrix);
        cudaFree(&this->deviceW);
        cudaFree(&this->deviceH);
        cudaGetLastError();
    }
    
    std::vector<float> CUDANetwork::retrieveOutput()
    {
        std::vector<float> out(this->outputSize);
        for (uint32_t i = 0; i < outputSize; i++)
        {
            memcpy(&out[i], this->outputs[i], sizeof(float));
        }
        return out;
    }

    void CUDANetwork::propagateInput()
    {
        dim3 threads(1024);

        uint32_t layerNumber = 0;
        for (auto const& layer : this->graphOrder)
        {
            dim3 blocksIz((layer.size() + threads.x - 1) / threads.x);
            if (layerNumber == 0)
            {   
                izhikevichInputLayerForward KERNEL2(blocksIz, threads)(this->deviceInputs, this->deviceV, this->deviceU, this->deviceIndexes[layerNumber], layer.size());
            }
            else
            {
                dim3 blocksSyn((connectionsIndexesNumber[layerNumber - 1] + threads.x - 1) / threads.x);
                synapse KERNEL2(blocksSyn, threads)(this->deviceV, this->deviceR,
                    this->deviceConnections2DArray, this->deviceConnectionsPointers, this->deviceIndexes[layerNumber],
                    this->deviceW[layerNumber - 1], this->deviceH[layerNumber - 1], this->connectionsIndexesNumber[layerNumber - 1],
                    this->deviceIMatrix[layerNumber - 1], layer.size());
                synapse_sum KERNEL2((layer.size() + this->block_x - 1) / this->block_x, nTPB)(this->deviceIMatrix[layerNumber - 1], this->deviceI[layerNumber - 1], layer.size(), graphOrder[layerNumber - 1].size());
                gpuErrchk();
                izhikevichLayerForward KERNEL2(blocksIz, threads)(this->deviceV, this->deviceU, this->deviceI[layerNumber - 1], this->deviceR, this->deviceConnections2DArray, this->deviceConnectionsPointers, this->deviceIndexes[layerNumber], layer.size());
            }
            gpuErrchk();

            layerNumber++;
        }
    }
}