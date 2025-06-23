#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>                              
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for matrix transpose
__global__ void transposeKernel(float* out, const float* in, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (x < cols && y < rows)
        out[x * rows + y] = in[y * cols + x];  // Transpose element
}

// CPU matrix transpose
void transposeCPU(std::vector<float>& out, const std::vector<float>& in, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[c * rows + r] = in[r * cols + c];
}

// Validate equality
bool validate(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
    for (size_t i = 0; i < a.size(); ++i)
        if (fabs(a[i] - b[i]) > epsilon)
            return false;
    return true;
}

// Print top-left corner of matrix
void printMatrixPreview(const std::vector<float>& mat, int rows, int cols, const std::string& label, int preview = 5) {
    std::cout << label << " (Top-Left " << preview << "x" << preview << "):\n";
    for (int r = 0; r < preview; ++r) {
        for (int c = 0; c < preview; ++c)
            std::cout << mat[r * cols + c] << "\t";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    const int rows = 1000;
    const int cols = 1000;
    const int size = rows * cols;

    std::vector<float> input(size);
    for (int i = 0; i < size; ++i)
        input[i] = static_cast<float>(i + 1);

    std::vector<float> cpu_result(size), gpu_result(size);

    // CPU Transpose
    auto start_cpu = std::chrono::high_resolution_clock::now();
    transposeCPU(cpu_result, input, rows, cols);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // GPU Transpose
    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    transposeKernel<<<gridSize, blockSize>>>(d_out, d_in, rows, cols);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;

    cudaMemcpy(gpu_result.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    // Print sample
    printMatrixPreview(input, rows, cols, "Original Matrix");
    printMatrixPreview(cpu_result, cols, rows, "CPU Transposed Matrix");
    printMatrixPreview(gpu_result, cols, rows, "GPU Transposed Matrix");

    // Validate and time results
    bool isValid = validate(cpu_result, gpu_result);
    std::cout << "Validation: " << (isValid ? "PASS" : "FAIL") << "\n";
    std::cout << "CPU Transpose Time: " << cpu_time.count() << " ms\n";
    std::cout << "GPU Transpose Time: " << gpu_time.count() << " ms\n";
    std::cout << "Speedup: " << (cpu_time.count() / gpu_time.count()) << "x\n";

    return 0;
}
