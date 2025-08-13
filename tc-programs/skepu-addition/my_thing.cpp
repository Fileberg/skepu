#include <skepu>
#include <iostream>
#include <skepu-lib/blas.hpp>
#include <skepu-lib/tcblas.hpp>
#include <thread>
#include <cstring>
#include <chrono>
#include <string>
#include <random>

#ifdef SKEPU_CUDA
#include <cuda.h>
#endif // SKEPU_CUDA



constexpr size_t SIZE = 64;
constexpr size_t M = 32;//SIZE;
constexpr size_t N = 16;//SIZE;
constexpr size_t K = 8;//SIZE;


template <typename T>
void print_matrix(const skepu::Matrix<T> mat) {
    for (size_t i{0}; i < mat.total_rows(); ++i) {
        for (size_t j{0}; j < mat.total_cols(); ++j) {
            std::cout << std::setprecision(5) << mat(i, j) << " ";
        }
        std::cout << std::endl;
    }
}


template <typename T>
void print_vector(const skepu::Vector<T> vec) {
    T const* vec_start = vec.data();
    for (size_t i{0}; i < vec.size(); ++i) {
        std::cout << std::setprecision(5) << vec_start[i] << " ";
    }
    std::cout << std::endl;
}


template <typename T>
void print_vector_short(const skepu::Vector<T> vec) {
    T const* vec_start = vec.data();
    std::cout << std::setprecision(3) << vec_start[0] << " ... " << vec_start[vec.size()-1] << std::endl;
}


void my_gemm() {
    skepu::Matrix<float> a(M, K, 1.f);
    skepu::Matrix<float> b(K, N, 1.f);
    skepu::Matrix<float> c1(M, N, 0.f);

    float *a_start = a.data();
    float *b_start = b.data();
    float *c1_start = c1.data();
    for (int i{0}; i < M; ++i) {
        for (int j{0}; j < K; ++j) {
            // (a_start)[i * K + j] = float(i * K + j);
            // (a_start)[i * N + j] = i == j ? 1.f : 0.f;
            // (b_start)[i * N + j] = i == j ? 1.f : 0.f;
            // (c1_start)[i * N + j] = 0.f;
        }
    }
    // a.randomizeReal(-1.0, 1.0);
    // b.randomizeReal(-1.0, 1.0);
    // c1.randomizeReal(-1.0, 1.0);
    skepu::Matrix<float> c2(c1);

    skepu::blas::Op no_trans = skepu::blas::Op::NoTrans;
    skepu::blas::Op trans = skepu::blas::Op::Trans;
    float alpha = 1.f;
    float beta = 0.f;

    std::cout << "pre:" << std::endl;
    std::cout << "a:" << std::endl;
    print_matrix(a);
    std::cout << "b:" << std::endl;
    print_matrix(b);
    std::cout << "c2:" << std::endl;
    print_matrix(c2);
    // std::cout << "c1:" << std::endl;
    // print_matrix(c1);

    // skepu::blas::gemm(
    //     no_trans,
    //     trans,
    //     M,
    //     N,
    //     K,
    //     alpha,
    //     a,
    //     M,
    //     b,
    //     K,
    //     beta,
    //     c1,
    //     M
    // );

    // std::cout << "====================================\npost c1: " << std::endl;
    // print_matrix(c1);

    skepu::tcblas::gemm(
        no_trans,
        no_trans,
        M,
        N,
        K,
        alpha,
        a,
        M,
        b,
        K,
        beta,
        c2,
        M
    );
    std::cout << "====================================\npost c2: " << std::endl;
    print_matrix(c2);
    // std::cout << "doñe" << std::endl;

    // bool no_error = true;
    // float *C1_start = c1.data();
    // float *C2_start = c2.data();
    // for (int i{0}; i < N; ++i) {
    //     for (int j{0}; j < M; ++j) {
    //         if (std::abs((C1_start)[i * N + j] - (C2_start)[i * N + j]) > 0.001) {
    //             std::cout << "erm wrong at " << i << ", " << j << " with values\nc1: " << (C1_start)[i * N + j] << "\nc2: " << (C2_start)[i * N + j] << "\nc1 - c2: " << (C1_start)[i * N + j] - (C2_start)[i * N + j] << std::endl;
    //             no_error = false;
    //             break;
    //         }
    //     }
    //     if (!no_error) {
    //         break;
    //     }
    // }
    // if (no_error) {
    //     std::cout << "c1 and c2 are the same practically!!!" << std::endl;
    // } else {
    //     std::cout << "====================================\npost c1: " << std::endl;
    //     print_matrix(c1);
    //     std::cout << "====================================\npost c2: " << std::endl;
    //     print_matrix(c2);
    // }
}


void my_gemv() {
    skepu::Matrix<float> A(M, N, 0.f);
    skepu::Vector<float> x(M, 1.f);
    skepu::Vector<float> y(N, 0.f);

    float *A_start = A.data();
    float *x_start = x.data();
    float *y_start = y.data();
    // A_start[0] = 1.f;
    // A_start[1] = 1.f;
    for (int i{0}; i < N; ++i) {
        A[i] = float(i);
    }
    // for (int i{0}; i < M; ++i) {
    //     for (int j{0}; j < N; ++j) {
    //         // (A_start)[i * N + j] = 1.f;
    //         (A_start)[i * N + j] = float(i * N + j);
    //         // (A_start)[i * N + j] = i == j ? i : 0.f;
    //         // (A_start)[i * N + j] = j == 0 ? i : 0.f;
    //     }
    //     // (x_start)[i] = 1.f;
    //     // (y_start)[i] = 1.f;
    // }
    // std::cout << "pre A:" << std::endl;
    // print_matrix(A);
    // std::cout << "pre x:" << std::endl;
    // print_vector(x);
    // std::cout << "pre y:" << std::endl;
    // print_vector(y);

    skepu::blas::Op no_trans = skepu::blas::Op::NoTrans;
    skepu::blas::Op trans = skepu::blas::Op::Trans;
    float alpha = 1.f;
    float beta = 1.f;

    // #ifdef SKEPU_CUDA
    // std::cout << y.isModified_CU(0) << std::endl;
    // #endif // SKEPU_CUDA

    skepu::tcblas::gemv(
        trans,
        M,
        N,
        alpha,
        A,
        M,
        x,
        1,
        beta,
        y,
        1
    );

    std::cout << "post A:" << std::endl;
    print_matrix(A);
    std::cout << "post x:" << std::endl;
    print_vector(x);
    std::cout << "post y:" << std::endl;
    print_vector(y);
    // std::cout << y(0) << " ... " << y(SIZE-1) << std::endl;

    // #ifdef SKEPU_CUDA
    // std::cout << y.isModified_CU(0) << std::endl;
    // #endif // SKEPU_CUDA
}


void my_dot() {
    skepu::Vector<float> x(N, 1.f);
    skepu::Vector<float> y(N, 1.f);

    // float *x_start = x.data();
    float *y_start = y.data();
    for (int i{0}; i < N; ++i) {
        // (x_start)[i] = 1.f;
        (y_start)[i] = float(i);
    }
    // x.randomizeReal(-1.0, 1.0);
    // y.randomizeReal(-1.0, 1.0);
    // std::cout << "pre A:" << std::endl;
    // print_matrix(A);
    // std::cout << "pre x:" << std::endl;
    // print_vector(x);
    // std::cout << "pre y:" << std::endl;
    // print_vector(y);

    // #ifdef SKEPU_CUDA
    // std::cout << y.isModified_CU(0) << std::endl;
    // #endif // SKEPU_CUDA

    auto answer = skepu::tcblas::dot(
        N,
        x,
        1,
        y,
        1
    );

    std::cout << "post x:" << std::endl;
    print_vector(x);
    std::cout << "post y:" << std::endl;
    print_vector(y);
    std::cout << "dot: " << answer << std::endl;
    // std::cout << y(0) << " ... " << y(SIZE-1) << std::endl;

    // #ifdef SKEPU_CUDA
    // std::cout << y.isModified_CU(0) << std::endl;
    // #endif // SKEPU_CUDA
}


#define gpuID 0
#define streamID 0
#define usePitch false
#define markOnlyLocalCopiesInvalid true

void thread_entry_gemv_tc(float* device_pointer_A, float* device_pointer_x, float* device_pointer_y, size_t const m, size_t const n, float const alpha, float const beta) {
    skepu::blas::Op no_trans = skepu::blas::Op::NoTrans;
// #ifndef SKEPU_NO_UPDATE_HOST
// #define SKEPU_NO_UPDATE_HOST
    skepu::tcblas::gemv(
        no_trans,
        m,
        n,
        alpha,
        device_pointer_A,
        m,
        device_pointer_x,
        1,
        beta,
        device_pointer_y,
        1
    );
// #endif // SKEPU_NO_UPDATE_HOST
}


float gemv_uf_notrans_noconj(float y, skepu::MatRow<float> Arow, skepu::Vec<float> x, skepu::blas::stride_type incx, float const alpha, skepu::blas::size_type const m, float const beta)
{
	float tmp = 0;
	for (uint i = 0; i < m; ++i)
        tmp += x(i * incx) * Arow(i);
	return beta * y + alpha * tmp;
}


void thread_entry_gemv_cuda(skepu::Matrix<float>* A, skepu::Vector<float>* x, skepu::Vector<float>* y, size_t const m, float const split, float const alpha, float const beta) {
    skepu::blas::Op no_trans = skepu::blas::Op::NoTrans;
    auto skel = skepu::Map<1>(gemv_uf_notrans_noconj);
    skel.setStride(1, 1);
    auto it_start = skepu::VectorIterator(*y, &y->data()[0]);
    auto it_end = skepu::VectorIterator(*y, &y->data()[(int)(m * split)]);
    skel(m, it_start, it_end, *y, *A, *x, 1, alpha, m, beta);
    // skepu::blas::gemv(
    //     no_trans,
    //     m,
    //     n,
    //     alpha,
    //     *A,
    //     m,
    //     *x,
    //     1,
    //     beta,
    //     *y,
    //     1
    // );
    // std::cout << "post y in cuda thread:" << std::endl;
    // print_vector(*y);
    // y->updateHost();
}


#ifdef SKEPU_CUDA
void gemv_hybrid_cuda_tc(skepu::Matrix<float>& A, skepu::Vector<float>& x, skepu::Vector<float>& y, int const m, int const n, int const split, float const alpha, float const beta) {
    double const cuda_split = split > 0 ? 1.f / split : (1.f + 1.f / split);
    skepu::AccessMode readMode = skepu::AccessMode::Read;
    skepu::AccessMode readWriteMode = skepu::AccessMode::ReadWrite;
    typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
    typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_x = x.updateDevice_CU(x.data(), x.size(), gpuID, readMode, usePitch, markOnlyLocalCopiesInvalid);
    typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_y = y.updateDevice_CU(y.data(), y.size(), gpuID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);

    float* A2 = (float *)(device_pointer_A->getDeviceDataPointer() + (int)(m * n * cuda_split));
    float* y_2 = (float *)(device_pointer_y->getDeviceDataPointer() + (int)(m * cuda_split));

    std::thread thread_split_cuda(thread_entry_gemv_cuda, &A, &x, &y, m, cuda_split, alpha, beta);
    std::thread thread_split_tc(thread_entry_gemv_tc, A2, (float *)device_pointer_x->getDeviceDataPointer(), y_2, (int)(m * (1.f - cuda_split)), n, alpha, beta);

    thread_split_cuda.join();
    thread_split_tc.join();

    device_pointer_y->changeDeviceData(true);
}
#endif // SKEPU_CUDA


void my_hybrid_gemv(int size, int const warmups, int const iterations, int const split, float const alpha, float const beta) {
    // size_t size = 1024 * 8;
    // size_t size = 32;
    // size_t split = 8;
    size_t m = size, n = size;
    skepu::Matrix<float> A(m, n);
    A.randomizeReal(-1, 2);
    skepu::Vector<float> x(n);
    // x.randomize(-1, 2);
    // skepu::Vector<float> y1(m, 0.f);
    // skepu::Vector<float> y2(m, 0.f);
    // skepu::Vector<float> y3(m, 0.f);
    skepu::Vector<float> y(m);
    // y.randomize(-1, 2);
    // skepu::Vector<float> y1(y);
    // skepu::Vector<float> y2(y);
    // skepu::Vector<float> y3(y);

    int seed = 2;
    double min = -1.0, max = 1.0;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min, max);
    // float *A_start = A.data();
    float *x_start = x.data();
    float *y_start = y.data();
    for (int i{0}; i < m; ++i) {
        for (int j{0}; j < n; ++j) {
            // (A_start)[i * n + j] = float(i * n + j);
        }
        (x_start)[i] = dis(gen);
        (y_start)[i] = dis(gen);
        // (x_start)[i] = 1.f;
        // (y_start)[i] = 1.f;
    }
    // skepu::Vector<float> yt(y);

#ifdef SKEPU_CUDA
    skepu::AccessMode readMode = skepu::AccessMode::Read;
    skepu::AccessMode readWriteMode = skepu::AccessMode::ReadWrite;
    // typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_y1 = y1.updateDevice_CU(y1.data(), y1.size(), gpuID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);

    int wrongs = 0;
    auto time_start = std::chrono::steady_clock::now();
    auto time_stop = std::chrono::steady_clock::now();

    if (split == 0) {
        // std::cout << "using -tc-" << std::endl;
        for (int i{0}; i < warmups; ++i) {
            // std::cout << "warming up nr " << i << std::endl;
            skepu::Vector<float> yc(y);
            typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
            typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_x = x.updateDevice_CU(x.data(), x.size(), gpuID, readMode, usePitch, markOnlyLocalCopiesInvalid);
            typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_yc = yc.updateDevice_CU(yc.data(), yc.size(), gpuID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
            device_pointer_yc->changeDeviceData(true);
            thread_entry_gemv_tc(device_pointer_A->getDeviceDataPointer(), device_pointer_x->getDeviceDataPointer(), device_pointer_yc->getDeviceDataPointer(), m, n, alpha, beta);
        }
        skepu::Vector<float> yc(y);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_x = x.updateDevice_CU(x.data(), x.size(), gpuID, readMode, usePitch, markOnlyLocalCopiesInvalid);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_yc = yc.updateDevice_CU(yc.data(), yc.size(), gpuID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
        device_pointer_yc->changeDeviceData(true);
        time_start = std::chrono::steady_clock::now();
        for (int i{0}; i < iterations; ++i) {
            // time_start = std::chrono::steady_clock::now();
            thread_entry_gemv_tc(device_pointer_A->getDeviceDataPointer(), device_pointer_x->getDeviceDataPointer(), device_pointer_yc->getDeviceDataPointer(), m, n, alpha, beta);
            cudaDeviceSynchronize();
            device_pointer_yc->changeDeviceData(true);
            // time_stop = std::chrono::steady_clock::now();
        }
        time_stop = std::chrono::steady_clock::now();

        bool wrong = false;

        if (!wrong)
            std::cout << "y using -tc- took: " << std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start).count() << std::endl;
    } else if (split == 1) {
        // std::cout << "using cuda" << std::endl;
        for (int i{0}; i < warmups; ++i) {
            skepu::Vector<float> yc(y);
            thread_entry_gemv_cuda(&A, &x, &yc, m, 1, alpha, beta);
        }
        skepu::Vector<float> yc(y);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_x = x.updateDevice_CU(x.data(), x.size(), gpuID, readMode, usePitch, markOnlyLocalCopiesInvalid);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_yc = yc.updateDevice_CU(yc.data(), yc.size(), gpuID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
        time_start = std::chrono::steady_clock::now();
        for (int i{0}; i < iterations; ++i) {
            // time_start = std::chrono::steady_clock::now();
            thread_entry_gemv_cuda(&A, &x, &yc, m, 1, alpha, beta);
            cudaDeviceSynchronize();
            // time_stop = std::chrono::steady_clock::now();
        }
        time_stop = std::chrono::steady_clock::now();

        bool wrong = false;
        if (!wrong)
            std::cout << "y using cuda took: " << std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start).count() << std::endl;
    } else {
        // std::cout << "using both" << std::endl;
        for (int i{0}; i < warmups; ++i) {
            skepu::Vector<float> yc(y);
            // typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_Cc = Cc.updateDevice_CU(Cc.data(), Cc.total_rows(), Cc.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
            // device_pointer_Cc->changeDeviceData(true);
            gemv_hybrid_cuda_tc(A, x, yc, m, n, split, alpha, beta);
        }
        skepu::Vector<float> yc(y);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_x = x.updateDevice_CU(x.data(), x.size(), gpuID, readMode, usePitch, markOnlyLocalCopiesInvalid);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_yc = yc.updateDevice_CU(yc.data(), yc.size(), gpuID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
        time_start = std::chrono::steady_clock::now();
        for (int i{0}; i < iterations; ++i) {
            // time_start = std::chrono::steady_clock::now();
            gemv_hybrid_cuda_tc(A, x, yc, m, n, split, alpha, beta);
            cudaDeviceSynchronize();
            // time_stop = std::chrono::steady_clock::now();
        }
        time_stop = std::chrono::steady_clock::now();

        bool wrong = false;
        if (!wrong)
            std::cout << "y using both took: " << std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start).count() << std::endl;
    }

#endif // SKEPU_CUDA

    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // std::cout << "Doñe" << std::endl;
    // for (int i{0}; i < m; ++i) {
    //     if (std::abs(y1(i) - y2(i)) > 0.001f) {
    //         std::cout << "But elem " << i << " was off!!! y1: " << y1(i) << ", y2: " << y2(i) << std::endl;
    //     }
    //     if (std::abs(y1(i) - y3(i)) > 0.001f) {
    //         std::cout << "But hybrid was wrong at elem " << i << ", y1: " << y1(i) << ", y3: " << y3(i) << std::endl;
    //         break;
    //     }
    // }

}



void thread_entry_gemm_tc(float* device_pointer_A, float* device_pointer_B, float* device_pointer_C, size_t const m, size_t const n, size_t const k, float const alpha, float const beta) {
    skepu::blas::Op no_trans = skepu::blas::Op::NoTrans;
// #ifndef SKEPU_NO_UPDATE_HOST
// #define SKEPU_NO_UPDATE_HOST

    skepu::tcblas::gemm(
        no_trans,
        no_trans,
        m,
        n,
        k,
        alpha,
        device_pointer_A,
        k,
        device_pointer_B,
        n,
        beta,
        device_pointer_C,
        n
    );
    // std::cout << "-tc- done" << std::endl;
// #endif // SKEPU_NO_UPDATE_HOST
}


// float gemm_uf_notransa_notransb(float c, const skepu::MatRow<float> Arow, const skepu::MatCol<float> Bcol, float alpha, float beta)
// {
// 	float res = 0;
// 	for (uint k = 0; k < Arow.cols; ++k)
// 		res += Arow(k) * Bcol(k);
// 	return alpha * res + beta * c;
// }
float gemm_uf_notransa_notransb(skepu::Index2D index, float c, const skepu::Mat<float> Arow, const skepu::Mat<float> Bcol, float alpha, float beta)
{
	float res = 0;
	for (uint k = 0; k < Arow.cols; ++k)
		res += Arow(index.row, k) * Bcol(k, index.col);
	return alpha * res + beta * c;
}


void thread_entry_gemm_cuda(skepu::Matrix<float>* A, skepu::Matrix<float>* B, skepu::Matrix<float>* C, size_t const m, size_t const n, double const split, float const alpha, float const beta) {
    // skepu::AccessMode readWriteMode = skepu::AccessMode::ReadWrite;
    // float alpha = 1.f;
    // float beta = 1.f;
// #ifndef SKEPU_NO_UPDATE_HOST
// #define SKEPU_NO_UPDATE_HOST
    auto skel = skepu::Map<1>(gemm_uf_notransa_notransb);
    skel.setBackend(skepu::Backend::Type::CUDA);
    // skel.setStride(0, 0);
    auto it_start = skepu::MatrixIterator<float>(C, &C->data()[0]);
    // auto it_start = skepu::MatrixIterator<float>(C, &C->data()[m * n / split]);
    auto it_end = skepu::MatrixIterator<float>(C, &C->data()[(int)(m * n * split)]);
    skel(m * n, it_start, it_end, *C, *A, *B, alpha, beta);
    // skel(it_end, *C, *A, *B, alpha, beta);
    // if (split > 1) {
    //     skel(*C, *C, *A, *B, alpha, beta);
    // } else {
    //     skel(*C, *C, *A, *B, alpha, beta);
    // }
    // std::cout << "cuda done" << std::endl;
// #endif // SKEPU_NO_UPDATE_HOST
// #ifdef SKEPU_CUDA
//     typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_C = C->updateDevice_CU(C->data(), C->total_rows(), C->total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
//     device_pointer_C->changeDeviceData(true);
// #endif // SKEPU_CUDA
//     C->updateHost();
}


#ifdef SKEPU_CUDA
template<typename T>
void gemm_hybrid_cuda_tc(skepu::Matrix<T>& A, skepu::Matrix<T>& B, skepu::Matrix<T>& C, float const alpha, float const beta, size_t const m, size_t const n, size_t const k, int const split) {
    // std::cout << "starting gemm hybrid cuda tc split: " << split << "\n";
    double const cuda_split = split > 0 ? 1.f / split : (1.f + 1.f / split);
    // std::cout << "cuda_split: " << cuda_split << "\n";
    skepu::AccessMode readMode = skepu::AccessMode::Read;
    skepu::AccessMode readWriteMode = skepu::AccessMode::ReadWrite;
    typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
    typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_B = B.updateDevice_CU(B.data(), B.total_rows(), B.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
    // std::cout << "getting C device data pointer\n";
    typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_C = C.updateDevice_CU(C.data(), C.total_rows(), C.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);

    float* A2 = (float *)(device_pointer_A->getDeviceDataPointer() + (int)(m * n * cuda_split));
    float* C_2 = (float *)(device_pointer_C->getDeviceDataPointer() + (int)(m * n * cuda_split));

    // auto time_start = std::chrono::steady_clock::now();

    // std::cout << "starting cuda thread\n";
    std::thread thread_split_cuda(thread_entry_gemm_cuda, &A, &B, &C, m, n, cuda_split, alpha, beta);

    // std::cout << "starting tc thread\n";
    std::thread thread_split_tc(thread_entry_gemm_tc, A2, device_pointer_B->getDeviceDataPointer(), C_2, (int)(m * (1.f - cuda_split)), n, k, alpha, beta);

    thread_split_cuda.join();
    // std::cout << "cuda thread done\n";

    thread_split_tc.join();

    // auto time_stop = std::chrono::steady_clock::now();
    // std::cout << "execution took: " << std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start).count() << std::endl;

    // std::cout << "tc thread done\n";
    device_pointer_C->changeDeviceData(true);
    // std::cout << "hybrid done\n";
}
#endif // SKEPU_CUDA


void my_hybrid_gemm(int size, int const warmups, int const iterations, int const split, float const alpha, float const beta) {
    // size_t size = 1024 * 1;
    // size_t size = 32;
    // size_t split = 8;
    size_t m = size, n = size, k = size;
    // std::cout << "creating A" << std::endl;
    skepu::Matrix<float> A(m, n);
    // std::cout << "randomizing A" << std::endl;
    A.randomizeReal(-1, 2);
    // std::cout << "creating B" << std::endl;
    skepu::Matrix<float> B(k, n);
    // std::cout << "randomizing B" << std::endl;
    B.randomizeReal(-1, 2);
    // std::cout << "creating C" << std::endl;
    skepu::Matrix<float> C(m, n, 0);
    // std::cout << "randomizing C" << std::endl;
    C.randomizeReal(-1, 2);
    // std::cout << "copying Ct" << std::endl;
    skepu::Matrix<float> Ct(C);

    // float *A_start = A.data();
    // float *B_start = B.data();
    // // float *C1_start = C1.data();
    // for (int i{0}; i < m; ++i) {
    //     for (int j{0}; j < n; ++j) {
    //         (A_start)[i * n + j] = 1.f;
    //         // (A_start)[i * n + j] = float(i * n + j);
    //         // (A_start)[i * n + j] = i == j ? i : 0.f;
    //         // (A_start)[i * n + j] = j == 0 ? i : 0.f;
    //         (B_start)[i * n + j] = i == 0 ? 1.f : 0.f;
    //         // (B_start)[i * n + j] = float(j * n + i);
    //     }
    // }

    // std::cout << "calculating" << std::endl;
    // for (int i{0}; i < iterations; ++i) {
    //     thread_entry_gemm_cuda(&A, &B, &Ct, m, n, 1, alpha, beta);
    // }
    // Ct.updateHost();
    // print_matrix(C);
    // thread_entry_gemm_tc(&A, &B, &Ct, m, n, 1);

#ifdef SKEPU_CUDA
    skepu::AccessMode readMode = skepu::AccessMode::Read;
    skepu::AccessMode readWriteMode = skepu::AccessMode::ReadWrite;

    int wrongs = 0;
    auto time_start = std::chrono::steady_clock::now();
    auto time_stop = std::chrono::steady_clock::now();

    if (split == 0) {
        // std::cout << "using -tc-" << std::endl;
        for (int i{0}; i < warmups; ++i) {
            // std::cout << "warming up nr " << i << std::endl;
            skepu::Matrix<float> Cc(C);
            typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
            typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_B = B.updateDevice_CU(B.data(), B.total_rows(), B.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
            typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_Cc = Cc.updateDevice_CU(Cc.data(), Cc.total_rows(), Cc.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
            device_pointer_Cc->changeDeviceData(true);
            thread_entry_gemm_tc(device_pointer_A->getDeviceDataPointer(), device_pointer_B->getDeviceDataPointer(), device_pointer_Cc->getDeviceDataPointer(), m, n, k, alpha, beta);
        }
        skepu::Matrix<float> Cc(C);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_B = B.updateDevice_CU(B.data(), B.total_rows(), B.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_Cc = Cc.updateDevice_CU(Cc.data(), Cc.total_rows(), Cc.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
        device_pointer_Cc->changeDeviceData(true);
        time_start = std::chrono::steady_clock::now();
        for (int i{0}; i < iterations; ++i) {
            // time_start = std::chrono::steady_clock::now();
            thread_entry_gemm_tc(device_pointer_A->getDeviceDataPointer(), device_pointer_B->getDeviceDataPointer(), device_pointer_Cc->getDeviceDataPointer(), m, n, k, alpha, beta);
            cudaDeviceSynchronize();
            device_pointer_Cc->changeDeviceData(true);
            // time_stop = std::chrono::steady_clock::now();
            // device_pointer_Cc->changeDeviceData(true);
        }
        time_stop = std::chrono::steady_clock::now();

        bool wrong = false;
        if (!wrong)
            std::cout << "C using -tc- took: " << std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start).count() << std::endl;
            // std::cout << "Ct: " << std::endl;
            // print_matrix(Ct);
            // std::cout << "Cc: " << std::endl;
            // print_matrix(Cc);
    } else if (split == 1) {
        // std::cout << "using cuda" << std::endl;
        for (int i{0}; i < warmups; ++i) {
            skepu::Matrix<float> Cc(C);
            thread_entry_gemm_cuda(&A, &B, &Cc, m, n, 1, alpha, beta);
        }
        skepu::Matrix<float> Cc(C);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_B = B.updateDevice_CU(B.data(), B.total_rows(), B.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
        typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_Cc = Cc.updateDevice_CU(Cc.data(), Cc.total_rows(), Cc.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
        time_start = std::chrono::steady_clock::now();
        for (int i{0}; i < iterations; ++i) {
            // time_start = std::chrono::steady_clock::now();
            thread_entry_gemm_cuda(&A, &B, &Cc, m, n, 1, alpha, beta);
            cudaDeviceSynchronize();
            // time_stop = std::chrono::steady_clock::now();
        }
        time_stop = std::chrono::steady_clock::now();

        bool wrong = false;
        if (!wrong)
            std::cout << "C using cuda took: " << std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start).count() << std::endl;
            // std::cout << "Ct: " << std::endl;
            // print_matrix(Ct);
            // std::cout << "Cc: " << std::endl;
            // print_matrix(Cc);
    } else {
        // std::cout << "using both" << std::endl;
        for (int i{0}; i < warmups; ++i) {
            skepu::Matrix<float> Cc(C);
            // typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_Cc = Cc.updateDevice_CU(Cc.data(), Cc.total_rows(), Cc.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
            // device_pointer_Cc->changeDeviceData(true);
            gemm_hybrid_cuda_tc(A, B, Cc, alpha, beta, m, n, k, split);
        }
        skepu::Matrix<float> Cc(C);
        time_start = std::chrono::steady_clock::now();
        for (int i{0}; i < iterations; ++i) {
            // skepu::Matrix<float> Cc(C);
            // typename skepu::backend::DeviceMemPointer_CU<float>* device_pointer_Cc = Cc.updateDevice_CU(Cc.data(), Cc.total_rows(), Cc.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);
            // device_pointer_Cc->changeDeviceData(true);

            // time_start = std::chrono::steady_clock::now();
            gemm_hybrid_cuda_tc(A, B, Cc, alpha, beta, m, n, k, split);
            cudaDeviceSynchronize();
            // time_stop = std::chrono::steady_clock::now();
            // device_pointer_Cc->changeDeviceData(true);
            // std::cout << "updating host\n";
        }
        time_stop = std::chrono::steady_clock::now();

        bool wrong = false;
        if (!wrong)
            std::cout << "C using both took: " << std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start).count() << std::endl;
    }

#endif // SKEPU_CUDA

    // std::cout << "Doñe" << std::endl;
    // for (int i{0}; i < m * n; ++i) {
    //     if (std::abs(C1(i) - C2(i)) > 0.001f) {
    //         std::cout << "But elem (" << int(i / n) << ", " << i % n << ") " << " was off!!! C1: " << C1(int(i / n), i % n) << ", C2: " << C2(int(i / n), i % n) << std::endl;
    //     }
    //     if (std::abs(C1(i) - C3(i)) > 0.001f) {
    //         std::cout << "But hybrid was wrong at elem (" << int(i / n) << ", " << i % n << ") " << ", C1: " << C1(int(i / n), i % n) << ", C3: " << C3(int(i / n), i % n) << std::endl;
    //     }
    // }
    // std::cout << "post A:" << std::endl;
    // print_matrix(A);
    // std::cout << "post B:" << std::endl;
    // print_matrix(B);
    // std::cout << "post C:" << std::endl;
    // print_matrix(C);
    // std::cout << "post C2:" << std::endl;
    // print_matrix(C2);
    // std::cout << "post C3:" << std::endl;
    // print_matrix(C3);
}



int main(int argc, char *argv[]) {
    int size, warmups, iterations, split;
    float alpha, beta;
    std::string program;
    if (argc > 7) {
        program = argv[1];
        size = std::atoi(argv[2]);
        warmups = std::atoi(argv[3]);
        iterations = std::atoi(argv[4]);
        split = argc > 4 ? std::atoi(argv[5]) : 2;
        alpha = argc > 5 ? std::atof(argv[6]) : 1.f;
        beta = argc > 6 ? std::atof(argv[7]) : 1.f;
    } else {
      std::cout << "usage: ./<exe> <gemm or gemv> <size> <warmups> <iterations> (optional <split> <alpha> <beta>)" << std::endl;
      return 0;
    }
    skepu::BackendSpec spec{"cuda"};
    skepu::setGlobalBackendSpec(spec);
    // int M = size, N = size, K = size;
    // my_gemm();
    // my_gemv();
    // my_dot();
    // std::cout << "running with size: " << size << ", warmups: " << warmups << ", iterations: " << iterations << ", split: " << split << ", alpha: " << alpha << std::endl;
    if (program == "gemm") {
        my_hybrid_gemm(size, warmups, iterations, split, alpha, beta);
    } else if (program == "gemv") {
        my_hybrid_gemv(size, warmups, iterations, split, alpha, beta);
    } else {
      std::cout << "usage: ./<exe> <gemm or gemv> <size> <warmups> <iterations> (optional <split> <alpha> <beta>)" << std::endl;
    }
    // std::cout << "Doñe\n";

    return 0;
}