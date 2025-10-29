
__global__ void skepu_skel_0_my_thing_precompiled_MapKernel_gemm_uf_notransa_notransb(float* skepu_output, skepu::PRNG::Placeholder,float *c, struct skepu::Mat<float> Arow, struct skepu::Mat<float> Bcol, float alpha, float beta,  size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base, skepu::StrideList<2> skepu_strides)
{
	size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_global_prng_id = skepu_i;
	size_t skepu_gridSize = blockDim.x * gridDim.x;
	float* skepu_Arow_base = Arow.data;
float* skepu_Bcol_base = Bcol.data;

	if (skepu_strides[0] < 0) { skepu_output += (-skepu_n + 1) * skepu_strides[0]; }
if (skepu_strides[1] < 0) { c += (-skepu_n + 1) * skepu_strides[1]; }


	while (skepu_i < skepu_n)
	{
		skepu::Index2D skepu_index;
skepu_index.row = (skepu_base + skepu_i) / skepu_w2;
skepu_index.col = (skepu_base + skepu_i) % skepu_w2;
		
		auto skepu_res = skepu_userfunction_skepu_skel_0skel_gemm_uf_notransa_notransb::CU(skepu_index, c[skepu_i * skepu_strides[1]], Arow, Bcol, alpha, beta);
		skepu_output[skepu_i * skepu_strides[0]] = skepu_res;
		skepu_i += skepu_gridSize;
	}
}
