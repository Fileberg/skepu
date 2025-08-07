#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <random>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

/*
run with
cmake .. && make && ./my_cublas
./my_cublas
*/



template <typename T>
void print_vector(const std::vector<T> vec) {
    T const* vec_start = vec.data();
    for (size_t i{0}; i < vec.size(); ++i) {
        std::cout << std::setprecision(3) << vec_start[i] << " ";
    }
    std::cout << std::endl;
}


using data_type = float;


// goal is to use fp32 btw
int main(int argc, char const **argv) {
  int size;
  int iterations;
  int warmups;
  float alpha;
  float beta;
  if (argc > 3) {
    size = std::atoi(argv[1]);
    iterations = std::atoi(argv[2]);
    warmups = std::atoi(argv[3]);
    alpha = argc > 4 ? std::atof(argv[4]) : 1.f;
    beta = argc > 5 ? std::atof(argv[5]) : 0.01f;
  } else {
    std::cout << "usage: ./<exe> <size> <iterations> <warmups> (optional <alpha> <beta>)" << std::endl;
    return 0;
  }
  int M = size, N = size, K = size, lda = size, ldb = size, ldc = size;
  // M = 8;
  // N = 4;
  // alpha = 1.f;
  // beta = 0.f;

  // create cublas things
  cublasHandle_t cublas_handle = NULL;
  cudaStream_t stream = NULL;

  cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);

  // init matrices
  int seed = 2;
  double min = -1.0, max = 1.0;
  std::vector<data_type> matrix_a_host(M * N);
  std::vector<data_type> matrix_b_host(K * N);
  std::vector<data_type> matrix_c_host(M * N);
  std::vector<data_type> matrix_1_host(M * N);
  std::vector<data_type> vector_x_host(N);
  std::vector<data_type> vector_y_host(M);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<data_type> dis(min, max);
  for (int i{0}; i < M; ++i) {
    for (int j{0}; j < N; ++j) {
      // matrix_a_host[j * M + i] = dis(gen);
      matrix_a_host[i * N + j] = float(i * N + j);
      // matrix_b_host[j * M + i] = dis(gen);
      // matrix_c_host[j * M + i] = dis(gen);
      // matrix_a_host[i * M + j] = i * size + j;
      // matrix_b_host[i * M + j] = i * size + j;
      // matrix_c_host[i * M + j] = i * size + j;
      // matrix_1_host[i * M + j] = 1;
      if(i == 0) vector_x_host[j] = data_type(1);
    }
    vector_y_host[i] = data_type(0);
  }

  data_type* matrix_a_device = nullptr;
  data_type* matrix_b_device = nullptr;
  data_type* matrix_c_device = nullptr;
  data_type* vector_x_device = nullptr;
  data_type* vector_y_device = nullptr;

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  cublasCreate(&cublas_handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(cublas_handle, stream);

  // copy to gpu
  cudaMalloc(reinterpret_cast<void **>(&matrix_a_device), sizeof(data_type) * matrix_a_host.size());
  cudaMalloc(reinterpret_cast<void **>(&matrix_b_device), sizeof(data_type) * matrix_b_host.size());
  cudaMalloc(reinterpret_cast<void **>(&matrix_c_device), sizeof(data_type) * matrix_c_host.size());
  cudaMalloc(reinterpret_cast<void **>(&vector_x_device), sizeof(data_type) * vector_x_host.size());
  cudaMalloc(reinterpret_cast<void **>(&vector_y_device), sizeof(data_type) * vector_y_host.size());

  cudaMemcpyAsync(matrix_a_device, matrix_a_host.data(), sizeof(data_type) * matrix_a_host.size(), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(matrix_b_device, matrix_b_host.data(), sizeof(data_type) * matrix_b_host.size(), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(matrix_c_device, matrix_c_host.data(), sizeof(data_type) * matrix_c_host.size(), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(vector_x_device, vector_x_host.data(), sizeof(data_type) * vector_x_host.size(), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(vector_y_device, vector_y_host.data(), sizeof(data_type) * vector_y_host.size(), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  cudaEvent_t events[2];
  cudaError_t error = cudaEventCreate(&events[0]);
  if (error != cudaSuccess) {
    std::cout << "Error creating first event!\n";
    return 1;
  }
  error = cudaEventCreate(&events[1]);
  if (error != cudaSuccess) {
    std::cout << "Error creating second event!\n";
    return 1;
  }

  cublasSgemv(cublas_handle, transa, M, N, &alpha, matrix_a_device, lda, vector_x_device, 1, &beta, vector_y_device, 1);
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
  cudaMemcpyAsync(matrix_a_host.data(), matrix_a_device, sizeof(data_type) * matrix_a_host.size(), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(vector_x_host.data(), vector_x_device, sizeof(data_type) * vector_x_host.size(), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(vector_y_host.data(), vector_y_device, sizeof(data_type) * vector_y_host.size(), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
  cudaMemcpy(vector_y_host.data(), vector_y_device, sizeof(data_type) * vector_y_host.size(), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // perform warmup operation
  for(int i{0}; i < warmups; ++i) {
    cublasSgemm(cublas_handle, transa, transb, M, N, K, &alpha, matrix_a_device, lda, matrix_b_device, ldb, &beta, matrix_c_device, ldc);
  }
  // re-copy to c since cuBLAS uses it as out
  cudaMemcpyAsync(matrix_c_device, matrix_c_host.data(), sizeof(data_type) * matrix_c_host.size(), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  error = cudaEventRecord(events[0]);
  if (error != cudaSuccess) {
    std::cout << "Error recording start event!\n";
    return 1;
  }

  // perform operation
  for(int i{0}; i < iterations; ++i) {
    cublasSgemm(cublas_handle, transa, transb, M, N, K, &alpha, matrix_a_device, lda, matrix_b_device, ldb, &beta, matrix_c_device, ldc);
    cudaMemcpyAsync(matrix_a_device, matrix_c_device, sizeof(data_type) * matrix_c_host.size(), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
}

  cudaMemcpyAsync(matrix_b_device, matrix_1_host.data(), sizeof(data_type) * matrix_b_host.size(), cudaMemcpyHostToDevice, stream);
  cudaMemsetAsync(matrix_c_device, 0, sizeof(data_type) * matrix_c_host.size(), stream);
  cudaStreamSynchronize(stream);
  cublasSgemm(cublas_handle, transa, transb, M, N, K, &alpha, matrix_a_device, lda, matrix_b_device, ldb, &beta, matrix_c_device, ldc);

  cudaMemcpyAsync(matrix_b_device, matrix_c_device, sizeof(data_type) * matrix_c_host.size(), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(matrix_a_device, matrix_1_host.data(), sizeof(data_type) * matrix_a_host.size(), cudaMemcpyHostToDevice, stream);
  cudaMemsetAsync(matrix_c_device, 0, sizeof(data_type) * matrix_c_host.size(), stream);
  cudaStreamSynchronize(stream);
  cublasSgemm(cublas_handle, transa, transb, M, N, K, &alpha, matrix_a_device, lda, matrix_b_device, ldb, &beta, matrix_c_device, ldc);

  error = cudaEventRecord(events[1]);
  if (error != cudaSuccess) {
    std::cout << "Error recording second event!\n";
    return 1;
  }
  error = cudaEventSynchronize(events[1]);
  if (error != cudaSuccess) {
    std::cout << "Error synchronizing second event!\n";
    return 1;
  }

  float runtime_ms{0};
  error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  if (error != cudaSuccess) {
    std::cout << "Error checking elapsed time!\n";
    return 1;
  }


  iterations += 2;
  size_t ma_ops = size_t(M) * N * K; // amount of multiply-add ops
  double tflops_s{ 2.0 * double(ma_ops) / (runtime_ms / double(iterations) / 1000.) / 1.0e12 }; // 2 flops per multiply-add op

  std::cout << iterations << " iterations run for: " << runtime_ms << " ms - avg is: " << double(runtime_ms) / double(iterations) << " - Tflops/s: " << tflops_s << std::endl;

  // cudaMemcpy(matrix_c_host.data(), matrix_c_device, sizeof(data_type) * matrix_c_host.size(), cudaMemcpyDeviceToHost);
  // for (int i{0}; i < size; ++i) {
  //   for (int j{0}; j < size; ++j) {
  //     std::cout << std::fixed << std::setprecision(3) << matrix_c_host[i * size + j] << "\t";
  //   }
  //   std::cout << std::endl;
  // }

  return 0;
}