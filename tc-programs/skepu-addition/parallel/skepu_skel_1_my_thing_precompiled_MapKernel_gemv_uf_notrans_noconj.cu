
__global__ void skepu_skel_1_my_thing_precompiled_MapKernel_gemv_uf_notrans_noconj(float* skepu_output, skepu::PRNG::Placeholder,float *y, struct skepu::MatRow<float> Arow, struct skepu::Vec<float> x, int incx, const float alpha, const unsigned long m, const float beta,  size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base, skepu::StrideList<2> skepu_strides)
{
	size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_global_prng_id = skepu_i;
	size_t skepu_gridSize = blockDim.x * gridDim.x;
	float* skepu_Arow_base = Arow.data;
float* skepu_x_base = x.data;

	if (skepu_strides[0] < 0) { skepu_output += (-skepu_n + 1) * skepu_strides[0]; }
if (skepu_strides[1] < 0) { y += (-skepu_n + 1) * skepu_strides[1]; }


	while (skepu_i < skepu_n)
	{
		
		Arow.data = skepu_Arow_base + skepu_i * Arow.cols;

		auto skepu_res = skepu_userfunction_skepu_skel_1skel_gemv_uf_notrans_noconj::CU(y[skepu_i * skepu_strides[1]], Arow, x, incx, alpha, m, beta);
		skepu_output[skepu_i * skepu_strides[0]] = skepu_res;
		skepu_i += skepu_gridSize;
	}
}
