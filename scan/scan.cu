#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include <iostream>
#include "CycleTimer.h"

#define THREADS_PER_BLOCK 1024


#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


/*
CUDA kernel
A naive version of exlcusive scan
*/
__global__ void work_inefficient_scan_kernel_1(int *X, int *Y, int InputSize){
    int idx = blockIdx.x*blockDim.x +threadIdx.x;
    int sum = 0;
    for(int i = 0; i < idx; i++ ){
        sum += X[i];
    }
    Y[idx] = sum;
}

/*
CUDA Kernel
An N*log(N) version of exclusive scan
However this version cannot spanning to different blocks
*/
__global__ void work_inefficient_scan_kernel_2(int *X, int *Y, int InputSize) {
    __shared__ int XY[THREADS_PER_BLOCK];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < InputSize) {XY[threadIdx.x] = X[i];}
      // the code below performs iterative scan on XY
    for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
        __syncthreads();
        int in1 = XY[threadIdx.x-stride];
        __syncthreads();
        XY[threadIdx.x] += in1;
    }
    Y[i] = XY[threadIdx.x] - X[i];
}
    
/*
CUDA Kernel for In-Block Exclusive Scan
Block size must be smaller than 1024 in 2080Ti
An O(N) version of exclusive scan for array of arbitary length
*/

__global__ void inblock_eff_scan(int *X, int *Y, int InputSize, int *FormerSum) {
    // XY[2*BLOCK_SIZE] is in shared memory
    __shared__ int XY[THREADS_PER_BLOCK * 2];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < InputSize) {
        XY[threadIdx.x] = X[i];
    }
    
      // the code below performs iterative scan on XY
    for (unsigned int stride = 1; stride <= THREADS_PER_BLOCK; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1; 
        if(index < InputSize)
            XY[index] += XY[index - stride];//index is alway bigger than stride
        __syncthreads();
    }
      // threadIdx.x+1 = 1,2,3,4....
      // stridek index = 1,3,5,7...

    for (unsigned int stride = THREADS_PER_BLOCK/2; stride > 0 ; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < InputSize)
            XY[index + stride] += XY[index];  
    }

    __syncthreads();
    if (i < InputSize){
        Y[i] = XY[threadIdx.x]-X[i]+*FormerSum;
        //printf("Final: Y[%d] = %d\n",i,Y[i]);
    }
    __syncthreads();
    if (i == InputSize-1){
        *FormerSum = X[i]+Y[i];
    }
}

void efficient_exclusive_scan(int *X, int *Y, int InputSize){
    std::cout<<InputSize<<std::endl;
    int *tmp = {0};
    cudaMalloc((void **)&tmp, sizeof(int));
    for(int i = 0; i < InputSize; i += THREADS_PER_BLOCK){
        const int threadsPerBlock = THREADS_PER_BLOCK;
        int len = (i+threadsPerBlock >= InputSize)? InputSize-i: threadsPerBlock;
        const int blocks = len / threadsPerBlock+1;
        inblock_eff_scan<<<blocks, threadsPerBlock>>>(&X[i],&Y[i],len,tmp);
        cudaCheckError( cudaDeviceSynchronize() ); 
    }
    cudaFree(tmp);
}

void test_inblock_exclusive_scan(int *X, int *Y, int InputSize){
    int *tmp = {0};
    cudaMalloc((void **)&tmp, sizeof(int));
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocks = InputSize / threadsPerBlock+1;
    int offset = 4;
    inblock_eff_scan<<<blocks, threadsPerBlock>>>(&X[offset],&Y[offset],InputSize-offset,tmp);
    cudaCheckError( cudaDeviceSynchronize() ); 
    cudaFree(tmp);
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel segmented scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep input
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    printf("Start ex_scan\n");
    efficient_exclusive_scan(input, result, N);

}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of segmented scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    return 0; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
