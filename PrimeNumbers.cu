#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// ================= CPU Sequential =================
bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i <= std::sqrt(n); i += 2)
        if (n % i == 0) return false;
    return true;
}

std::vector<int> sequential_primes(int limit) {
    std::vector<int> primes;
    for (int i = 2; i <= limit; ++i)
        if (is_prime(i))
            primes.push_back(i);
    return primes;
}

// ================= Multithreaded =================
void find_primes_range(int start, int end, std::vector<int>& local_primes) {
    for (int i = start; i <= end; ++i)
        if (is_prime(i))
            local_primes.push_back(i);
}

std::vector<int> parallel_primes(int limit, int num_threads) {
    std::vector<std::thread> threads;
    std::vector<std::vector<int>> thread_primes(num_threads);
    int chunk = limit / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk + 2;
        int end = (t == num_threads - 1) ? limit : (t + 1) * chunk + 1;
        threads.emplace_back(find_primes_range, start, end, std::ref(thread_primes[t]));
    }

    for (auto& th : threads)
        th.join();

    std::vector<int> primes;
    for (const auto& v : thread_primes)
        primes.insert(primes.end(), v.begin(), v.end());

    return primes;
}

// ================= CUDA =================
__device__ bool is_prime_device(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i <= sqrtf((float)n); i += 2)
        if (n % i == 0) return false;
    return true;
}

__global__ void find_primes_kernel(int limit, int* d_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > limit) return;
    if (is_prime_device(idx)) {
        d_primes[idx] = 1;
    }
}

std::vector<int> cuda_primes(int limit) {
    int* d_primes;
    int* h_primes = new int[limit + 1]();

    cudaMalloc(&d_primes, (limit + 1) * sizeof(int));
    cudaMemset(d_primes, 0, (limit + 1) * sizeof(int));

    int threads_per_block = 256;
    int blocks = (limit + threads_per_block) / threads_per_block;

    find_primes_kernel<<<blocks, threads_per_block>>>(limit, d_primes);
    cudaDeviceSynchronize();

    cudaMemcpy(h_primes, d_primes, (limit + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> primes;
    for (int i = 2; i <= limit; ++i) {
        if (h_primes[i]) primes.push_back(i);
    }

    cudaFree(d_primes);
    delete[] h_primes;
    return primes;
}

// ================= Main =================
int main() {
    const int limit = 10'000'000;
    const int num_threads = std::thread::hardware_concurrency();

    std::cout << "Finding primes up to " << limit << "...\n\n";

    // Sequential
    auto t1 = std::chrono::high_resolution_clock::now();
    auto primes_seq = sequential_primes(limit);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_seq = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    // Multithreaded
    auto t3 = std::chrono::high_resolution_clock::now();
    auto primes_par = parallel_primes(limit, num_threads);
    auto t4 = std::chrono::high_resolution_clock::now();
    auto ms_par = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

    // CUDA
    auto t5 = std::chrono::high_resolution_clock::now();
    auto primes_cuda = cuda_primes(limit);
    auto t6 = std::chrono::high_resolution_clock::now();
    auto ms_cuda = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();

    // Output
    std::cout << "Sequential    : " << primes_seq.size() << " primes in " << ms_seq << " ms\n";
    std::cout << "Multithreaded : " << primes_par.size() << " primes in " << ms_par << " ms\n";
    std::cout << "CUDA          : " << primes_cuda.size() << " primes in " << ms_cuda << " ms\n";

    std::cout << "\nSpeedups vs Sequential:\n";
    std::cout << "Multithreaded: " << static_cast<float>(ms_seq) / ms_par << "x\n";
    std::cout << "CUDA         : " << static_cast<float>(ms_seq) / ms_cuda << "x\n";

    return 0;
}