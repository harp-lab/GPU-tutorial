/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


#include "device_launch_parameters.h"

#include <time.h>
#include <chrono> 
#include <iostream> 
using namespace std;
using namespace std::chrono;

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
                                                        float *B, int wA,
                                                        int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

int matrixMultiplyWithoutCUDA(float* C, float* A, float* B, const dim3& dimsA, const dim3& dimsB) {
    //printf("Row of A: %d, Column of A: %d\n", dimsA.y, dimsA.x);
    //printf("Row of B: %d, Column of B: %d\n", dimsB.y, dimsB.x);

    int x = dimsA.y; // row of A
    int y = dimsA.x; // column of A and row of B
    int z = dimsB.x; // column of B
                     // So, row and column of C is x & z, respectively

    //x = 6;
    //y = 4;
    //z = 5;

    for (int i = 0; i < x; i++) {
        for (int j = 0; j < z; j++) {
            int sum = 0;
            for (int k = 0; k < y; k++) {
                int actualIndexOfA = i * y + k;
                int actualIndexOfB = z * k + j;
                //printf("(A: %d, B: %d), ", actualIndexOfA, actualIndexOfB);
                sum = sum + (A[actualIndexOfA] * B[actualIndexOfB]);
            }
            int actualIndexOfC = i * z + j;
            //printf("(Stored at C: %d\n", actualIndexOfC);
            C[actualIndexOfC] = sum;
        }
    }
    //printf("End of Without CUDA\n");
    return 0;
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv,
                   int block_size, const dim3 &dimsA,
                   const dim3 &dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = reinterpret_cast<float *>(malloc(mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = reinterpret_cast<float *>(malloc(mem_size_B));

    // Initialize host memory
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = reinterpret_cast<float *>(malloc(mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    //printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_C, d_A, d_B,
                                                dimsA.x, dimsB.x);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_C, d_A, d_B,
                                                dimsA.x, dimsB.x);
    }

    //printf("done\n");

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    //cudaEvent_t start;
    //checkCudaErrors(cudaEventCreate(&start));

    //cudaEvent_t stop;
    //checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    //checkCudaErrors(cudaEventRecord(start, NULL));

    // Execute the kernel
    int nIter = 1; // was 300

    auto start1 = high_resolution_clock::now();
    
    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16> <<< grid, threads >>>(d_C, d_A, d_B,
                                                    dimsA.x, dimsB.x);
        } else {
            MatrixMulCUDA<32> <<< grid, threads >>>(d_C, d_A, d_B,
                                                    dimsA.x, dimsB.x);
        }
    }

    cudaDeviceSynchronize();

    auto stop1 = high_resolution_clock::now();

    auto duration1 = duration_cast<microseconds>(stop1 - start1);

    cout << duration1.count() << ", ";

    // Record the stop event
    //checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    //checkCudaErrors(cudaEventSynchronize(stop));

    //float msecTotal = 0.0f;
    //checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    //float msecPerMatrixMul = msecTotal / nIter;
    //double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
    //                           static_cast<double>(dimsA.y) *
    //                           static_cast<double>(dimsB.x);
    //double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
    //                   (msecPerMatrixMul / 1000.0f);
    //printf(
    //    "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops," \
    //    " WorkgroupSize= %u threads/block\n",
    //    gigaFlops,
    //    msecPerMatrixMul,
    //    flopsPerMatrixMul,
    //    threads.x * threads.y);

    //printf("CUDA= %.3f msec,", msecPerMatrixMul);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    //printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6;  // machine zero

    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    //printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");


    // Allocate CUDA events that we'll use for timing
    //cudaEvent_t start2;
    //checkCudaErrors(cudaEventCreate(&start2));

    //cudaEvent_t stop2;
    //checkCudaErrors(cudaEventCreate(&stop2));

    // Record the start event
    //checkCudaErrors(cudaEventRecord(start2, NULL));
    
    auto start2 = high_resolution_clock::now();

    matrixMultiplyWithoutCUDA(h_C, h_A, h_B, dimsA, dimsB);
    
    auto stop2 = high_resolution_clock::now();

    
    auto duration2 = duration_cast<microseconds>(stop2 - start2);
    
    cout << duration2.count() << endl;


    // Record the stop event
    //checkCudaErrors(cudaEventRecord(stop2, NULL));

    // Wait for the stop event to complete
    //checkCudaErrors(cudaEventSynchronize(stop2));

    //float msc2 = 0.0f;
    //checkCudaErrors(cudaEventElapsedTime(&msc2, start2, stop2));


    //printf("Without CUDA= %.3f msec\n", msc2);




    
    
    
    
    
    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    printf("\nNOTE: The CUDA Samples are not meant for performance"\
           "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv) {
    //printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices" \
               " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    //int dev = findCudaDevice(argc, (const char **)argv); // This line prints the available CUDA compatible GPU information

    int matrix_result;

    int block_size = 32;
    
    bool alter = true;
    for (int x = 1, y = 1; x < 1024 && y < 1024; ) {
        
        dim3 dimsA(5 * 2 * block_size, x * block_size, 1);
        dim3 dimsB(y * block_size, 5 * 2 * block_size, 1);

        // width of Matrix A
        if (checkCmdLineFlag(argc, (const char**)argv, "wA")) {
            dimsA.x = getCmdLineArgumentInt(argc, (const char**)argv, "wA");
        }

        // height of Matrix A
        if (checkCmdLineFlag(argc, (const char**)argv, "hA")) {
            dimsA.y = getCmdLineArgumentInt(argc, (const char**)argv, "hA");
        }

        // width of Matrix B
        if (checkCmdLineFlag(argc, (const char**)argv, "wB")) {
            dimsB.x = getCmdLineArgumentInt(argc, (const char**)argv, "wB");
        }

        // height of Matrix B
        if (checkCmdLineFlag(argc, (const char**)argv, "hB")) {
            dimsB.y = getCmdLineArgumentInt(argc, (const char**)argv, "hB");
        }

        if (dimsA.x != dimsB.y) {
            printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
                dimsA.x, dimsB.y);
            exit(EXIT_FAILURE);
        }

        printf("MatrixA(%d,%d), MatrixB(%d,%d), ", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

        matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);



        if (alter) {
            x = x * 2;
        }
        else {
            y = y * 2;
        }
        alter = !alter;
    }

    

    //exit(matrix_result);
    return 3;
}

