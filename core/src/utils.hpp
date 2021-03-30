#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>

#ifdef __INTELLISENSE__
void __syncthreads();  // workaround __syncthreads warning
#define KERNEL2(grid, block)
#define KERNEL3(grid, block, sh_mem)
#define KERNEL4(grid, block, sh_mem, stream)
#else
#define KERNEL2(grid, block) <<< grid, block >>>
#define KERNEL3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#endif

#define gpuErrCodechk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuErrchk() { gpuAssert(__FILE__, __LINE__); }
inline void gpuAssert(const char* file, int line, bool abort = true)
{
    auto code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}