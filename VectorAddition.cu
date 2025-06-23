#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip> // For formatted output (C++17)

// CUDA kernel for parallel vector addition
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Sequential CPU implementation of vector addition
void vectorAddCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < A.size(); ++i)
        C[i] = A[i] + B[i];
}

// Validate that two vectors are nearly equal
bool validate(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
    for (int i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > epsilon)
            return false;
    return true;
}

void runVectorAdditionTest(const std::vector<float>& A, const std::vector<float>& B) {
    int N = A.size();
    std::vector<float> C_cpu(N);
    std::vector<float> C_gpu(N);

    // Sequential CPU timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(A, B, C_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // GPU timing
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vectorAddKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;

    // Copy result back to CPU
    cudaMemcpy(C_gpu.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Compare results
    bool ok = validate(C_cpu, C_gpu);

    // Print results
    std::cout << "\nVector Size: " << N << "\n";
    std::cout << "CPU Time: " << cpu_time.count() << " ms\n";
    std::cout << "GPU Time: " << gpu_time.count() << " ms\n";
    std::cout << "Speedup:  " << (cpu_time.count() / gpu_time.count()) << "x\n";
    std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << "\n";

    // Print a few elements for small vectors
    if (N <= 16) {
        std::cout << "\nSample Output (Vector Add):\n";
        for (int i = 0; i < N; ++i)
            std::cout << std::fixed << std::setprecision(2)
                      << A[i] << " + " << B[i] << " = " << C_gpu[i] << "\n";
    }
}

int main() {
    std::cout << "Testing Vector Addition using C++17 and CUDA 11.x\n";

    // Small test vector
    std::vector<float> A1 = {1.5, 3.0, 5.2, 7.1, 0.5, -1.0, 2.2, 4.4};
    std::vector<float> B1 = {2.5, -3.0, 1.8, 0.9, 3.5, 1.0, -2.2, 5.6};
    runVectorAdditionTest(A1, B1);

    // Medium test vector
    int mid_size = 10000;
    std::vector<float> A2(mid_size, 1.0f);
    std::vector<float> B2(mid_size, 2.0f);
    runVectorAdditionTest(A2, B2);

    // Large test vector
    int large_size = 1000000;
    std::vector<float> A3(large_size, 3.0f);
    std::vector<float> B3(large_size, 4.0f);
    runVectorAdditionTest(A3, B3);

    return 0;
}
