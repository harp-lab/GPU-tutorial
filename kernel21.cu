
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono> 
using namespace std;
using namespace std::chrono;

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size, const int blk, const int thrd);
void addWithoutCuda(int* d, const int* a, const int* b, unsigned int size);
void printArray(int* a, const int arraySize, bool line = false);
bool verify(int* c, int* d, const int arraySize);


__global__ void addKernel(int* c, const int* a, const int* b, const int* helper)
{
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int totalThread = helper[0];
    int seqChunk = helper[1];

    int st = seqChunk * (blockId * totalThread + threadId);

    for (int i = 0;i < seqChunk;i++) {
        c[i + st] = a[i + st] * b[i + st];
    }
    //c[i] = a[i] * b[i];
}

int main()
{
    int rangeStart = 1500;
    int rangeEnd = 2500;
    const int numOfBlocks = 32;
    const int numOfThreads = 64;
    const int arraySizesLength = 21;
    int arraySizes[arraySizesLength] = { 2048,4096,8192,16384,32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 268435456 * 2, 268435456 * 3, 268435456 * 4};//,436207616};
    printf("no.,arraySize,seqRuntimeAvg,parRuntimeAvg,seqRuntimeFirst,parRuntimeFirst,seqRuntimeBest,parRuntimeBest,seqRuntimeWorst,parRuntimeWorst,status\n");
    for (int itrSz = 0; itrSz < arraySizesLength; itrSz++) {
        const int arraySize = arraySizes[itrSz];
        int batch = 5;
        int* seqRuntime = new int[batch];
        int* parRuntime = new int[batch];
        int seqRuntimeFirst = 0;
        int seqRuntimeMax = 0;
        int seqRuntimeMin = 0;
        int seqRuntimeTotal = 0;
        int parRuntimeFirst = 0;
        int parRuntimeMax = 0;
        int parRuntimeMin = 0;
        int parRuntimeTotal = 0;
        bool batchCheck = true;
        int itr = 0;
        for (; itr < batch; itr++) {

            bool checkFinal = true; //A check to verify


            int* a = new int[arraySize];
            //a[0] = 11;
            //a[1] = 22;
            //a[2] = 33;
            //a[3] = 44;
            //a[4] = 55;
            for (int i = 0; i < arraySize; i++) {
                int randomNumber = rangeStart + (rangeEnd - rangeStart) * (rand() / (RAND_MAX + 1.0));
                a[i] = randomNumber;
                //randomNumber >> a[i];//a[i] = rand();//
            }
            //const int a[arraySize] = { 1, 2, 3, 4, 5 };

            int* b = new int[arraySize];
            //b[0] = 10;
            //b[1] = 20;
            //b[2] = 30;
            //b[3] = 40;
            //b[4] = 50;
            for (int i = 0; i < arraySize; i++) {
                int randomNumber = 1 + 100 * (rand() / (RAND_MAX + 1.0));
                b[i] = randomNumber;
                //randomNumber >> a[i];//a[i] = rand();//
            }
            //    const int b[arraySize] = { 10, 20, 30, 40, 50 };
            int* c = new int[arraySize]; //Will be populated parallelly
            //int c[arraySize] = { 0 };
            int* d = new int[arraySize]; //Will be populated sequentially
            // Add vectors sequentially.
            auto start1 = high_resolution_clock::now();
            addWithoutCuda(d, a, b, arraySize);
            auto stop1 = high_resolution_clock::now();

            // Add vectors in parallel.
            auto start2 = high_resolution_clock::now();
            cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize, numOfBlocks, numOfThreads);
            auto stop2 = high_resolution_clock::now();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "addWithCuda failed!");
                return 1;
            }

            //printArray(a, arraySize);
            //printf(" * ");
            //printArray(b, arraySize);
            //printf(" = ");
            //printArray(c, arraySize, true);

            checkFinal = verify(c, d, arraySize);
            if (!checkFinal) {
                batchCheck = checkFinal;
            }
            //printf("%s\n", checkFinal ? "true" : "false");

            auto duration1 = duration_cast<microseconds>(stop1 - start1);
            auto duration2 = duration_cast<microseconds>(stop2 - start2);
            seqRuntime[itr] = duration1.count();
            parRuntime[itr] = duration2.count();
            if (itr == 0) {
                seqRuntimeFirst = seqRuntime[0];
                seqRuntimeMax = seqRuntime[0];
                seqRuntimeMin = seqRuntime[0];
                parRuntimeFirst = parRuntime[0];
                parRuntimeMax = parRuntime[0];
                parRuntimeMin = parRuntime[0];
            }
            else {
                if (seqRuntimeMax < seqRuntime[itr])
                    seqRuntimeMax = seqRuntime[itr];
                else if (seqRuntimeMin > seqRuntime[itr])
                    seqRuntimeMin = seqRuntime[itr];
                if (parRuntimeMax < parRuntime[itr])
                    parRuntimeMax = parRuntime[itr];
                else if (parRuntimeMin > parRuntime[itr])
                    parRuntimeMin = parRuntime[itr];
            }
            seqRuntimeTotal += seqRuntime[itr];
            parRuntimeTotal += parRuntime[itr];

            //printf("%d;%d\n", seqRuntimeTotal, parRuntimeTotal);

            //printf("{1,2,3,4,5} * {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
            //    c[0], c[1], c[2], c[3], c[4]);

            // cudaDeviceReset must be called before exiting in order for profiling and
            // tracing tools such as Nsight and Visual Profiler to show complete traces.
            cudaStatus = cudaDeviceReset();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceReset failed!");
                return 1;
            }
            free(a);
            free(b);
            free(c);
            free(d);

            //printf("Itr: %d, Batch: %d\n", itr, batch);
        }

        int seqRuntimeAvg = seqRuntimeTotal / batch;
        int parRuntimeAvg = parRuntimeTotal / batch;
        //printf("no.,arraySize,seqRuntimeAvg,parRuntimeAvg,seqRuntimeFirst,parRuntimeFirst,seqRuntimeBest,parRuntimeBest,seqRuntimeWorst,parRuntimeWorst,status\n");
        printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n", (itrSz + 1), arraySize, seqRuntimeAvg, parRuntimeAvg, seqRuntimeFirst, parRuntimeFirst, seqRuntimeMin, parRuntimeMin, seqRuntimeMax, parRuntimeMax, batchCheck ? "true" : "false");
        free(seqRuntime);
        free(parRuntime);
    }


    
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size, const int blk, const int thrd)
{
    int seqChunk = size / (blk * thrd);
    int* helper = new int(2);
    helper[0] = thrd;
    helper[1] = seqChunk;
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    int* dev_helper = 0;


    cudaError_t cudaStatus;
    //auto startCudaMallocMemCpy = high_resolution_clock::now();

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! for dev_c");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! for dev_a");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! for dev_b");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_helper, 2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! for dev_helper");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_helper, helper, 2 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    //auto stopCudaMallocMemCpy = high_resolution_clock::now();

    // Launch a kernel on the GPU with one thread for each element.
    

    addKernel << <blk, thrd >> > (dev_c, dev_a, dev_b, dev_helper);

    
    
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    //auto stopKernel = high_resolution_clock::now();
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    //auto stopCpyBck = high_resolution_clock::now();
Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_helper);


    //auto stopMemFree = high_resolution_clock::now();
    
    //auto durationMallocMemCpy = duration_cast<microseconds>(stopCudaMallocMemCpy - startCudaMallocMemCpy);
    //auto durationKernel = duration_cast<microseconds>(stopKernel - stopCudaMallocMemCpy);
    //auto durationCpyBck = duration_cast<microseconds>(stopCpyBck - stopKernel);
    //auto durationMemFree = duration_cast<microseconds>(stopMemFree - stopCpyBck);
    //auto totalGPU = durationMallocMemCpy + durationKernel + durationCpyBck + durationMemFree;

    //printf("GPU Breakdown\nCPU to GPU,%d,Kernel,%d,GPU to CPU,%d,Freeing up GPU,%d,Total,%d\n", durationMallocMemCpy, durationKernel, durationCpyBck, durationMemFree, totalGPU);

    return cudaStatus;
}

void addWithoutCuda(int* d, const int* a, const int* b, unsigned int size) {
    for (int i = 0;i < size;i++)
        d[i] = a[i] * b[i];
}

void printArray(int* a, const int arraySize, bool line) {
    for (int i = 0;i < arraySize;i++) {
        if (i == 0)
            printf("{%d,", a[i]);
        else if (i == arraySize - 1)
            printf("%d}", a[i]);
        else
            printf("%d,", a[i]);
    }
    if (line)
        printf("\n");
}
bool verify(int* c, int* d, const int arraySize) {
    for (int i = 0;i < arraySize;i++) {
        if (c[i] != d[i])
            return false;
    }
    return true;
}
