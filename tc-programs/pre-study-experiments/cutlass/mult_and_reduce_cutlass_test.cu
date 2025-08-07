#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cutlass/cutlass.h>
// #include <cutlass/numeric_types.h>
// #include <cutlass/core_io.h>

#include <cutlass/gemm/device/gemm.h>

// #include "cutlass/util/command_line.h"
#include <cutlass/util/host_tensor.h>

// #include "cutlass/util/reference/device/gemm.h"
// #include "cutlass/util/reference/host/tensor_norm.h"
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
// #include "cutlass/util/tensor_view_io.h"

#include <vector>
// #include "../my_util.h"


/*
run with:
make && ./my_cutlass
./my_cutlass
*/

using data_type = float;

using RowMajor = cutlass::layout::RowMajor;
using ColMajor = cutlass::layout::ColumnMajor;

using my_gemm = cutlass::gemm::device::Gemm<
  data_type,
  RowMajor,
  data_type,
  RowMajor,
  data_type,
  RowMajor,
  data_type,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 16>,
  cutlass::gemm::GemmShape<64, 64, 16>,
  cutlass::gemm::GemmShape<16, 8, 8>,
  cutlass::epilogue::thread::LinearCombination<
    data_type,
    128 / cutlass::sizeof_bits<data_type>::value,
    data_type,
    data_type>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  4,
  4,
  false
  // ,cutlass::arch::OpMultiplyAddFastF32
  ,cutlass::arch::OpMultiplyAdd
>;


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
  int M = size, N = size, K = size;

  // init matrices
  int seed = 2;
  double min = -1.0, max = 1.0;
  std::vector<data_type> matrix_a_host(M * K);
  std::vector<data_type> matrix_b_host(K * N);
  std::vector<data_type> matrix_c_host(M * N);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<data_type> dis(min, max);
  for (int i{0}; i < M; ++i) {
    for (int j{0}; j < N; ++j) {
      matrix_a_host[i * M + j] = dis(gen);
      matrix_b_host[i * M + j] = dis(gen);
      matrix_c_host[i * M + j] = dis(gen);
      // matrix_a_host[i * M + j] = i * size + j;
      // matrix_b_host[i * M + j] = i * size + j;
      // matrix_c_host[i * M + j] = i * size + j;
    }
  }

  // init cutlass matrices
  cutlass::gemm::GemmCoord problem_size({M, N, K});

  cutlass::HostTensor<data_type, RowMajor> matrix_a(problem_size.mk());
  cutlass::HostTensor<data_type, RowMajor> matrix_b(problem_size.kn());
  cutlass::HostTensor<data_type, RowMajor> matrix_c(problem_size.mn());
  cutlass::HostTensor<data_type, RowMajor> matrix_d(problem_size.mn());

  matrix_a.copy_in_host_to_host(matrix_a_host.data());
  matrix_b.copy_in_host_to_host(matrix_b_host.data());
  matrix_c.copy_in_host_to_host(matrix_c_host.data());

  // copy to gpu
  matrix_a.sync_device();
  matrix_b.sync_device();
  matrix_c.sync_device();

  typename my_gemm::Arguments my_args{
    problem_size,
    matrix_a.device_ref(),
    matrix_b.device_ref(),
    matrix_c.device_ref(),
    matrix_d.device_ref(),
    {alpha, beta},
    1
  };

  size_t my_workspace_size = my_gemm::get_workspace_size(my_args);

  cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

  my_gemm gemm_op;

  cutlass::Status my_status = gemm_op.can_implement(my_args);
  my_status = gemm_op.initialize(my_args, my_workspace.get());

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

  // warmups before actual execution
  for(int i{0}; i < warmups; ++i) {
    my_status = gemm_op();
    // matrix_c.sync_host();
  }

  error = cudaEventRecord(events[0]);
  if (error != cudaSuccess) {
    std::cout << "Error recording start event!\n";
    return 1;
  }

  // time to test for real
  for(int i{0}; i < iterations; ++i) {
    my_status = gemm_op();
    matrix_a.copy_in_device_to_device(matrix_d.device_data());
  }

  // put C in A and 1 in B and 0 in C
  // matrix_a.copy_in_device_to_device(matrix_c.device_data());
  cutlass::reference::device::TensorFill(matrix_b.device_view(), 1.f);
  cutlass::reference::device::TensorFill(matrix_c.device_view(), 0.f);
  my_status = gemm_op();

  // // put C in B and 1 in A and 0 in C
  matrix_b.copy_in_device_to_device(matrix_d.device_data());
  cutlass::reference::device::TensorFill(matrix_a.device_view(), 1.f);
  cutlass::reference::device::TensorFill(matrix_c.device_view(), 0.f);
  my_status = gemm_op();

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

  std::cout << iterations << " iterations run for:\t" << runtime_ms << " ms - avg is:\t" << double(runtime_ms) / double(iterations) << " - Tflops:\t" << tflops_s << std::endl;


  // for printing to see if the result is correct!
  // matrix_d.sync_host();
  // std::cout << "============D============" << std::endl;
  // for (int i{0}; i < M; ++i) {
  //   for (int j{0}; j < N; ++j) {
  //     std::cout << std::fixed << std::setprecision(3) << matrix_d.at({i, j}) << "\t";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << "============~============" << std::endl;

  // std::vector<data_type>matrix_reference_host(M * N);
  // float sum{0};
  // for (int i{0}; i < size; ++i) {
  //   for (int j{0}; j < size; ++j) {
  //     matrix_reference_host[i * size + j] = beta * matrix_c_host[i * size + j];
  //     for (int k{0}; k < size; ++k) {
  //       matrix_reference_host[i * size + j] += alpha * matrix_a_host[i * size + k] * matrix_b_host[k * size + j];
  //     }
  //   }
  // }
  // std::cout << "===========REF===========" << std::endl;
  // for (int i{0}; i < size; ++i) {
  //   for (int j{0}; j < size; ++j) {
  //     sum += matrix_reference_host[i * size + j];
  //     std::cout << std::fixed << std::setprecision(3) << matrix_reference_host[i * size + j] << "\t";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << sum << std::endl;

  return 0;
}