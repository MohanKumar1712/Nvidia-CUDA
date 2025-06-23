#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <cassert>
#include <cuda_runtime.h>

const int BIN_COUNT = 256;

// ------------------------ Data Generation ------------------------
std::vector<uint8_t> generate_data(size_t size) {
    std::vector<uint8_t> data(size);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& val : data)
        val = dist(rng);
    return data;
}

// ------------------------ Sequential Histogram ------------------------
std::vector<int> compute_histogram_sequential(const std::vector<uint8_t>& data) {
    std::vector<int> histogram(BIN_COUNT, 0);
    for (uint8_t val : data)
        ++histogram[val];
    return histogram;
}

// ------------------------ Multithreaded Histogram ------------------------
std::vector<int> compute_histogram_parallel(const std::vector<uint8_t>& data, int num_threads) {
    std::vector<std::vector<int>> local_histograms(num_threads, std::vector<int>(BIN_COUNT, 0));
    std::vector<std::thread> threads;
    size_t chunk_size = data.size() / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads - 1) ? data.size() : start + chunk_size;
            for (size_t i = start; i < end; ++i)
                ++local_histograms[t][data[i]];
        });
    }

    for (auto& th : threads) th.join();

    std::vector<int> final_histogram(BIN_COUNT, 0);
    for (int b = 0; b < BIN_COUNT; ++b)
        for (int t = 0; t < num_threads; ++t)
            final_histogram[b] += local_histograms[t][b];

    return final_histogram;
}

// ------------------------ CUDA Kernel ------------------------
__global__ void histogram_kernel(const uint8_t* data, int* histogram, size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    atomicAdd(&histogram[data[tid]], 1);
}

// ------------------------ CUDA Wrapper ------------------------
std::vector<int> compute_histogram_cuda(const std::vector<uint8_t>& data) {
    size_t size = data.size();
    uint8_t* d_data;
    int* d_histogram;
    std::vector<int> h_histogram(BIN_COUNT, 0);

    cudaMalloc(&d_data, size * sizeof(uint8_t));
    cudaMalloc(&d_histogram, BIN_COUNT * sizeof(int));

    cudaMemcpy(d_data, data.data(), size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, BIN_COUNT * sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    histogram_kernel<<<blocks, threadsPerBlock>>>(d_data, d_histogram, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_histogram.data(), d_histogram, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_histogram);

    return h_histogram;
}

// ------------------------ Verifier ------------------------
void compare_histograms(const std::vector<int>& h1, const std::vector<int>& h2, const std::string& label) {
    assert(h1.size() == h2.size());
    for (size_t i = 0; i < h1.size(); ++i) {
        if (h1[i] != h2[i]) {
            std::cerr << "❌ Mismatch at bin " << i << ": " << h1[i] << " != " << h2[i] << " [" << label << "]\n";
            return;
        }
    }
    std::cout << "" << label << " matches the sequential version.\n";
}

// ------------------------ Main ------------------------
int main() {
    size_t data_size = 1'000'000'000;
    int num_threads = std::thread::hardware_concurrency();

    std::cout << "Generating data...\n";
    auto data = generate_data(data_size);

    // Sequential
    auto t1 = std::chrono::high_resolution_clock::now();
    auto h_seq = compute_histogram_sequential(data);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_seq = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    // Multithreaded
    auto t3 = std::chrono::high_resolution_clock::now();
    auto h_par = compute_histogram_parallel(data, num_threads);
    auto t4 = std::chrono::high_resolution_clock::now();
    auto ms_par = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

    // CUDA
    auto t5 = std::chrono::high_resolution_clock::now();
    auto h_cuda = compute_histogram_cuda(data);
    auto t6 = std::chrono::high_resolution_clock::now();
    auto ms_cuda = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();

    // Output results
    std::cout << "\nTiming (on " << data_size << " elements):\n";
    std::cout << "Sequential    : " << ms_seq  << " ms\n";
    std::cout << "Multithreaded : " << ms_par  << " ms\n";
    std::cout << "CUDA          : " << ms_cuda << " ms\n";

    std::cout << "\nSpeedups:\n";
    std::cout << "Multithreaded : " << (float)ms_seq / ms_par << "x\n";
    std::cout << "CUDA          : " << (float)ms_seq / ms_cuda << "x\n";

    // Validate correctness
    compare_histograms(h_seq, h_par, "Multithreaded");
    compare_histograms(h_seq, h_cuda, "CUDA");
    
    
    return 0;
}