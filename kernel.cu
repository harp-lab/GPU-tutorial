#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <chrono> 
#include <iostream> 
using namespace std;
using namespace std::chrono;

void print(int* array, unsigned int size, string name);

__global__ void power(int *c, const int *a, const int *helper)
{
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int numberOfThread = helper[0];
    int seqPart = helper[1];
    int st = seqPart * (blockId*numberOfThread + threadId);
    
    
    int i = 0;
    for (;i < seqPart;i++) {
        c[st+i] = (a[st+i] * a[st+i]);
        printf("A[%d]: %d & C[%d]: %d\n", (st + i),a[st + i],(st + i),c[st + i]);
    }
}

cudaError_t squareWithCuda(int* c, int* a, int size, int blk, int thrd)
{
    print(a, size, "A at the beginning of the CUDA method");
    int seqPart = size / (blk * thrd);
    int* helper = new int(2);
    helper[0] = thrd;
    helper[1] = seqPart;
    int *dev_helper = 0;
    int *dev_a = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_helper, 2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed while allocating spaces for the helper!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed while allocating spaces for output!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed while allocating spaces for input!");
        goto Error;
    }
    print(a, size, "A just before the kernel call");
    cudaStatus = cudaMemcpy(dev_helper, helper, 2 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed while transfering values of the helper!");
        goto Error;
    }
    
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed while transfering values of a (input)!");
        goto Error;
    }
    
    // Launch a kernel
    power <<< blk, thrd >> > (dev_c, dev_a, dev_helper);
    

        cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_helper);
    cudaFree(dev_c);
    cudaFree(dev_a);
    //cudaFree(dev_b);
    
    return cudaStatus;
}

void squareWithoutCuda(int* d, int* a, int size) {
    for (int i = 0;i < size;i++) {
        d[i] = a[i] * a[i];
        //a[i] * a[i] >> d[i];//[i] = a[i] * a[i];//
    }
}

int main()
{
    int numOfBlocks = 1;
    int numOfThreads = 1;
    bool checkFinal = true;
    int arraySize = 1024;
    //long* a = new long(arraySize);
    int* c = new int(arraySize);
    int* d = new int(arraySize);
    int* a = new int(arraySize);
    //int a[arraySize];
    //int c[arraySize] = { 0 };
    //int d[arraySize] = { 0 };
    //for (int itr = 1;itr <= 50;itr++) {        
        
        for (int i = 0; i < arraySize; i++) {
            int randomNumber = 1 + 100 * (rand() / (RAND_MAX + 1.0));
            a[i] = randomNumber;
            //randomNumber >> a[i];//a[i] = rand();//
        }
        print(a, arraySize, "Array a");
        auto start1 = high_resolution_clock::now();
        squareWithoutCuda(d, a, arraySize);
        auto stop1 = high_resolution_clock::now();
        print(d, arraySize, "Seq");
        auto start2 = high_resolution_clock::now();
        squareWithCuda(c, a, arraySize, numOfBlocks, numOfThreads);
        auto stop2 = high_resolution_clock::now();
        print(c, arraySize, "Par");
        
        

        auto duration1 = duration_cast<microseconds>(stop1 - start1);
        
        auto duration2 = duration_cast<microseconds>(stop2 - start2);
        cout << numOfBlocks << "," << numOfThreads << "," << duration1.count() << "," << duration2.count() << endl;

        bool check = true;
        int count = 0;
        for (int i = 0;i < arraySize;i++) {
            if (c[i] != d[i]) {
                check = false;
                break;
            }
            else {
                count++;
            }

        }
        delete[] a;
        delete[] c;
        delete[] d;
        if (!check) {
            checkFinal = !checkFinal;
        }
        
    //}
    if (checkFinal) {
        printf("%s\n==================\n", "PASS");
    }
    else {
        printf("%s\n==================\n", "FAIL");
    }

    return 0;
}
void print(int* array, unsigned int size, string name) {
    printf("%s","Array printing starts: ");
    cout << name << endl;
    for (int i = 0;i < size-1;i++) {
        printf("%d; ",array[i]);
    }
    printf("%d\n%s\n", array[size-1],"Array printing ends.");
}