#define SKEPU_PRECOMPILED 1
#define SKEPU_CUDA 1
#include <skepu>
#include <iostream>
#include <skepu-lib/blas.hpp>
/* BEGIN BLAS.HPP INJECTION */
//startOfBlasHPP;

#include <skepu-lib/complex.hpp>

#define BLAS_CONST // const

namespace cplx = skepu::complex;
using FComplex = cplx::complex<float>;
using DComplex = cplx::complex<double>;


// Can't have types defined in an enclosed namespace (yet)
// Can't have type templates in SkepU (yet)
struct iamax_tmp_float
{
	float val;
	size_t index;
};

struct iamax_tmp_double
{
	double val;
	size_t index;
};

template<typename T>
struct iamax_tmp_type_helper {};

template<>
struct iamax_tmp_type_helper<float>
{
	using type = iamax_tmp_float;
};

template<>
struct iamax_tmp_type_helper<double>
{
	using type = iamax_tmp_double;
};

template<typename T>
using iamax_type = typename iamax_tmp_type_helper<T>::type;





namespace skepu {

namespace blas {
	
	using stride_type = int;
	using size_type = size_t;
	
#define SKEPU_BLAS_STRIDE_TYPE_UF int
#define SKEPU_BLAS_SIZE_TYPE_UF size_t
	
	// for zero types
	template< typename... Types >
	struct scalar_type_traits;

	// define scalar_type<> type alias
	template< typename... Types >
	using scalar_type = typename scalar_type_traits< Types... >::type;

	// for one type
	template< typename T >
	struct scalar_type_traits< T >
	{
	  using type = typename std::decay<T>::type;
	};

	// for two types
	// relies on type of ?: operator being the common type of its two arguments
	template< typename T1, typename T2 >
	struct scalar_type_traits< T1, T2 >
	{
		using type = typename std::decay< decltype( true ? std::declval<T1>() : std::declval<T2>() ) >::type;
	};
	
	// for either or both complex,
	// find common type of associated real types, then add complex
	template<> struct scalar_type_traits<FComplex, float> { using type = FComplex; };
	template<> struct scalar_type_traits<FComplex, double> { using type = DComplex; };
	template<> struct scalar_type_traits<DComplex, float> { using type = DComplex; };
	template<> struct scalar_type_traits<DComplex, double> { using type = DComplex; };
	
	template<> struct scalar_type_traits<float, FComplex> { using type = FComplex; };
	template<> struct scalar_type_traits<double, FComplex> { using type = DComplex; };
	template<> struct scalar_type_traits<float, DComplex> { using type = DComplex; };
	template<> struct scalar_type_traits<double, DComplex> { using type = DComplex; };

	// for three or more types
	template< typename T1, typename T2, typename... Types >
	struct scalar_type_traits< T1, T2, Types... >
	{
	    using type = scalar_type< scalar_type< T1, T2 >, Types... >;
	};
	
	
	
	
	// for zero types
	template< typename... Types >
	struct real_type_traits;

	// define real_type<> type alias
	template< typename... Types >
	using real_type = typename real_type_traits< Types... >::real_t;

	// define complex_type<> type alias
	template< typename... Types >
	using complex_type = skepu::complex::complex< real_type< Types... > >;

	// for one type
	template< typename T >
	struct real_type_traits<T>
	{
	    using real_t = T;
	};

	// for one complex type, strip complex
	template< typename T >
	struct real_type_traits< skepu::complex::complex<T> >
	{
	    using real_t = T;
	};

	// for two or more types
	template< typename T1, typename... Types >
	struct real_type_traits< T1, Types... >
	{
	    using real_t = scalar_type< real_type<T1>, real_type< Types... > >;
	};


	
	using namespace skepu::complex;

// ================================================================================
// ================================================================================
// ==========================       LEVEL 1 BLAS
// ================================================================================
// ================================================================================



// ----------------------------------------------------------
//   SWAP
// ----------------------------------------------------------


template<typename TX, typename TY>
skepu::multiple<TY, TX> swap_uf(TX x, TY y)
{
	return skepu::ret(y, x);
}

template<typename TX, typename TY>
void swap(
	size_type                        n,
	Vector<TX> &                     x,
	stride_type                      incx,
	Vector<TY> &                     y,
	stride_type                      incy
)
{
	// if (incx == incy == 1) AND (TX == TY) then pointer-swap
	auto skel = Map<2>(swap_uf<TX, TY>);
	skel.setStride(incx, incy, incx, incy);
	skel(x, y, x, y);
}


// ----------------------------------------------------------
//   SCAL
// ----------------------------------------------------------

template<typename TX, typename TS>
TX scal_uf(TX x, TS alpha)
{
	return x * alpha;
}

template<typename TX, typename TS>
void scal(
	size_type                        n,
	TS                               alpha,
	Vector<TX> &                     x,
	stride_type                      incx
)
{
	auto skel = Map<1>(scal_uf<TX, TS>);
	skel.setStride(incx, incx);
	skel(x, x, alpha);
}


// ----------------------------------------------------------
//   COPY
// ----------------------------------------------------------

template<typename TX, typename TY>
TY copy_uf(TX x)
{
	return x;
}

template<typename TX, typename TY>
void copy(
	size_type                        n,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	Vector<TY> &                     y,
	stride_type                      incy
)
{
	auto skel = Map<1>(copy_uf<TX, TY>);
	skel.setStride(incy, incx);
	skel(y, x);
}


// ----------------------------------------------------------
//   AXPY
// ----------------------------------------------------------

template<typename TX, typename TY, typename TA = scalar_type<TX, TY>>
TY axpy_uf(TX x, TY y, TA alpha)
{
	return alpha * x + y;
}

template<typename TX, typename TY, typename TS = scalar_type<TX, TY>>
void axpy(
	size_type                        n,
	TS                               alpha,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	Vector<TY> &                     y,
	stride_type                      incy
)
{
	auto skel = Map<2>(axpy_uf<TX, TY>);
	skel.setStride(incy, incx, incy);
	skel(y, x, y, alpha);
}


// ----------------------------------------------------------
//   DOT
// ----------------------------------------------------------

template<typename TX, typename TY, typename TR = scalar_type<TX, TY>>
TR dot_uf_1(TX x, TY y)
{
	return skepu::complex::conj(x) * y;
}

template<typename T>
T dot_uf_2(T lhs, T rhs)
{
	return lhs + rhs;
}

template<typename TX, typename TY>
scalar_type<TX, TY> dot(
	size_type                        n,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	Vector<TY> BLAS_CONST&           y,
	stride_type                      incy
)
{
	auto skel = MapReduce<2>(dot_uf_1<TX, TY>, dot_uf_2<scalar_type<TX, TY>>);
	skel.setStride(incx, incy);
	return skel(x, y);
}



// ----------------------------------------------------------
//   DOTU
// ----------------------------------------------------------

template<typename TX, typename TY, typename TR = scalar_type<TX, TY>>
TR dotu_uf_1(TX x, TY y)
{
	return x * y;
}

template<typename TX, typename TY>
scalar_type<TX, TY> dotu(
	size_type                        n,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	Vector<TY> BLAS_CONST&           y,
	stride_type                      incy
)
{
	auto skel = MapReduce<2>(dotu_uf_1<TX, TY>, dot_uf_2<scalar_type<TX, TY>>); // ???
	skel.setStride(incx, incy);
	return skel(x, y);
}



// ----------------------------------------------------------
//   NRM2
// ----------------------------------------------------------

template<typename T, typename TR = real_type<T>>
TR nrm2_uf_1(T x)
{
	return real(x) * real(x) + imag(x) * imag(x);
}

template<typename T>
T nrm2_uf_2(T lhs, T rhs)
{
	return lhs + rhs;
}

template<typename T>
real_type<T> nrm2(
	size_type                        n,
	Vector<T> BLAS_CONST&            x,
	stride_type                      incx
)
{
	auto skel = MapReduce<1>(nrm2_uf_1<T>, nrm2_uf_2<real_type<T>>);
	skel.setStride(incx);
	return sqrt(skel(x));
}


// ----------------------------------------------------------
//   ASUM
// ----------------------------------------------------------

template<typename T>
real_type<T> asum_uf_1(T x)
{
	return abs1(x);
}

template<typename T>
T asum(
	size_type                        n,
	Vector<T> BLAS_CONST&            x,
	stride_type                      incx
)
{
	auto skel = MapReduce<1>(asum_uf_1<T>, dot_uf_2<real_type<T>>);
	skel.setStride(incx);
	return skel(x);
}


// ----------------------------------------------------------
//   IAMAX
// ----------------------------------------------------------

template<typename T, typename R = real_type<T>, typename H = iamax_type<R>>
H iamax_uf_1(Index1D index, T xi)
{
	H tmp;
	tmp.val = abs1(xi);
	tmp.index = index.i;
	return tmp;
}

template<typename H>
H iamax_uf_2(H lhs, H rhs)
{
	return (lhs.val > rhs.val) ? lhs : rhs;
}

template<typename T>
size_type iamax(
	size_type                        n,
	Vector<T> BLAS_CONST&            x,
	stride_type                      incx
)
{
	auto skel = MapReduce<1>(iamax_uf_1<T>, iamax_uf_2<iamax_type<real_type<T>>>);
	skel.setStride(incx);
	return skel(x).index;
}


// ----------------------------------------------------------
//   ROTG
// ----------------------------------------------------------

template<typename T>
T sign(T a, T b)
{
	T x = (a >= 0 ? a : - a);
	return b >= 0 ? x : -x;
}

template<typename T>
void rotg(T *a, T *b, T *c, T *s)
{
	const T c_b4 = 1.;
  T d__1, d__2;
  T r, scale, z, roe;

  roe = *b;
  if (abs(*a) > abs(*b))
		roe = *a;
  
  scale = abs(*a) + abs(*b);
  
	if (scale != 0.)
	{
		/* Computing 2nd power */
	  d__1 = *a / scale;
		/* Computing 2nd power */
	  d__2 = *b / scale;
	  r = scale * sqrt(d__1 * d__1 + d__2 * d__2);
	  r = sign(c_b4, roe) * r;
	  *c = *a / r;
	  *s = *b / r;
	  z = 1.;
	  if (abs(*a) > abs(*b))
			z = *s;
	  
		if (abs(*b) >= abs(*a) && *c != 0.)
			z = 1. / *c;
			
	  *a = r;
	  *b = z;
		
	  return;
  }
	
  *c = 1.;
  *s = 0.;
  r = 0.;
  z = 0.;
	*a = r;
  *b = z;
	
  return;
}


// ----------------------------------------------------------
//   ROT
// ----------------------------------------------------------

template<typename TX, typename TY, typename TS = scalar_type<TX, TY>>
skepu::multiple<TX, TY> rot_uf(TX x, TY y, TS c, TS s)
{
	TS new_x = c * x + s * y;
	TS new_y = c * y - s * x;
	return skepu::ret(new_x, new_y);
}

template<typename TX, typename TY, typename TS = scalar_type<TX, TY>>
void rot(
	size_type                        n,
	Vector<TX> &                     x,
	stride_type                      incx,
	Vector<TY> &                     y,
	stride_type                      incy,
	TS                               c,
	TS                               s
)
{
  auto skel = Map<2>(rot_uf<TX, TY>);
	skel.setStride(incx, incy, incx, incy);
	skel(x, y, x, y, c, s);
}


// ----------------------------------------------------------
//   ROTMG
// ----------------------------------------------------------

// TODO implement

// ----------------------------------------------------------
//   ROTM
// ----------------------------------------------------------

// TODO implement




// ================================================================================
// ================================================================================
// ==========================       LEVEL 2 BLAS
// ================================================================================
// ================================================================================


// ----------------------------------------------------------
//   GEMV
// ----------------------------------------------------------

enum class Op     : char { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' };
enum class Uplo   : char { Upper    = 'U', Lower    = 'L', General   = 'G' };
enum class Diag   : char { NonUnit  = 'N', Unit     = 'U' };
enum class Side   : char { Left     = 'L', Right    = 'R' };


template<typename TA , typename TX , typename TY, typename TS = scalar_type<TA, TX, TY>>
TY gemv_uf_notrans_noconj(TY y, MatRow<TA> Arow, Vec<TX> x, stride_type incx, TS alpha, size_type m) // TODO: reference for proxies
{
	TS tmp = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF i = 0; i < m; ++i)
		y += x(i * incx) * Arow(i); // TODO negative incx
	return y + alpha * tmp;
}

template<typename TA , typename TX , typename TY, typename TS = scalar_type<TA, TX, TY>>
TY gemv_uf_trans_noconj(Index1D index, TY y, MatCol<TA> Acol, Vec<TX> x, stride_type incx, TS alpha, size_type m) // TODO: reference for proxies
{
	SKEPU_BLAS_SIZE_TYPE_UF j = index.i;
	TS tmp = alpha * x(j * incx); // TODO negative incx
	for (SKEPU_BLAS_SIZE_TYPE_UF i = 0; i < m; ++i)
		y += tmp * Acol(i);
	return y;
}

template<typename TA , typename TX , typename TY, typename TS = scalar_type<TA, TX, TY>>
TY gemv_uf_trans_conj(Index1D index, TY y, MatCol<TA> Acol, Vec<TX> x, stride_type incx, TS alpha, size_type m) // TODO: reference for proxies
{
	SKEPU_BLAS_SIZE_TYPE_UF j = index.i;
	TS tmp = alpha * x(j * incx); // TODO negative incx
	for (SKEPU_BLAS_SIZE_TYPE_UF i = 0; i < m; ++i)
		y += tmp * skepu::complex::conj(Acol(i));
	return y;
}


template<typename TA, typename TX, typename TY, typename TS = scalar_type<TA, TX, TY>>
void gemv(
	blas::Op 	                       trans,
	size_type                        m,
	size_type                        n,
	TS                               alpha,
	Matrix<TA> BLAS_CONST&           A,
	size_type                        lda,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	TS                               beta,
	Vector<TY> &                     y,
	stride_type                      incy
)
{
	const TS zero = 0, one  = 1;

  // quick return
  if (m == 0 || n == 0 || (alpha == zero && beta == one))
    return;
	
//  int64_t lenx = (trans == Op::NoTrans ? n : m);
//  int64_t leny = (trans == Op::NoTrans ? m : n);
//  int64_t kx = (incx > 0 ? 0 : (-lenx + 1)*incx);
//  int64_t ky = (incy > 0 ? 0 : (-leny + 1)*incy);
	
  // form y = beta*y
  if (beta != one)
		scal(m, beta, y, incy);
	
  if (alpha == zero)
    return;
	
  if (trans == Op::NoTrans)
	{
    auto skel = Map<1>(gemv_uf_notrans_noconj<TA, TX, TY>);
		skel.setStride(incy, incy);
		skel(y, y, A, x, incx, alpha, m);
  }
  else if (trans == Op::Trans)
	{
		auto skel = Map<1>(gemv_uf_trans_noconj<TA, TX, TY>);
		skel.setStride(incy, incy);
		skel(y, y, A, x, incx, alpha, m);
  }
  else // trans == Op::ConjTrans
	{
		auto skel = Map<1>(gemv_uf_trans_conj<TA, TX, TY>);
		skel.setStride(incy, incy);
		skel(y, y, A, x, incx, alpha, m);
  }
}


// ----------------------------------------------------------
//   GER
// ----------------------------------------------------------

template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
TA ger_uf(TX x, TY y, TS alpha)
{
	return alpha * x * conj(y);
}

template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
void ger(
	size_type                         m,
	size_type                         n,
	TS                                alpha,
	Vector<TX> BLAS_CONST&            x,
	stride_type                       incx,
	Vector<TY> BLAS_CONST&            y,
	stride_type                       incy,
	Matrix<TA>&                       A,
	stride_type                       lda 
)
{
	auto skel = skepu::MapPairs<1, 1>(ger_uf<TX, TY, TA>);
	// skel.setStride(incx, incy); // TODO implement
	skel.setInPlace(true);
	skel(A, x, y, alpha);
}


// ----------------------------------------------------------
//   GERU
// ----------------------------------------------------------

template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
TA geru_uf(TX x, TY y, TS alpha)
{
	return alpha * x * y;
}

template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
void geru(
	size_type                         m,
	size_type                         n,
	TS                                alpha,
	Vector<TX> BLAS_CONST&            x,
	stride_type                       incx,
	Vector<TY> BLAS_CONST&            y,
	stride_type                       incy,
	Matrix<TA>&                       A,
	stride_type                       lda 
)
{
	auto skel = skepu::MapPairs<1, 1>(geru_uf<TX, TY, TA>);
	// skel.setStride(incx, incy); // TODO implement
	skel.setInPlace(true);
	skel(A, x, y, alpha);
}





// ================================================================================
// ================================================================================
// ==========================       LEVEL 3 BLAS
// ================================================================================
// ================================================================================


// ----------------------------------------------------------
//   GEMM
// ----------------------------------------------------------

template<typename T1, typename T2>
T1 gemm_uf_matrix_scale(T1 x, T2 beta)
{
	return x * beta;
}

template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf_notransa_notransb(TC c, const skepu::MatRow<TA> Arow, const skepu::MatCol<TB> Bcol, TS alpha, TS beta)
{
	TC res = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF k = 0; k < Arow.cols; ++k)
		res += Arow(k) * Bcol(k);
	return alpha * res + beta * c;
}

template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf_notransa_transb(TC c, const skepu::MatRow<TA> Arow, const skepu::MatRow<TB> Brow, TS alpha, TS beta)
{
	TC res = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF k = 0; k < Arow.cols; ++k)
		res += Arow(k) * Brow(k);
	return alpha * res + beta * c;
}

template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf_notransa_transconjb(TC c, const skepu::MatRow<TA> Arow, const skepu::MatRow<TB> Brow, TS alpha, TS beta)
{
	TC res = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF k = 0; k < Arow.cols; ++k)
		res += Arow(k) * conj(Brow(k));
	return alpha * res + beta * c;
}


template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf_transa_notransb(TC c, const skepu::MatCol<TA> Acol, const skepu::MatCol<TB> Bcol, TS alpha, TS beta)
{
	TC res = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF k = 0; k < Acol.rows; ++k)
		res += Acol(k) * Bcol(k);
	return alpha * res + beta * c;
}

template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf_transa_transb(TC c, const skepu::MatCol<TA> Acol, const skepu::MatRow<TB> Brow, TS alpha, TS beta)
{
	TC res = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF k = 0; k < Acol.rows; ++k)
		res += Acol(k) * Brow(k);
	return alpha * res + beta * c;
}

template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf_transa_transconjb(TC c, const skepu::MatCol<TA> Acol, const skepu::MatCol<TB> Brow, TS alpha, TS beta)
{
	TC res = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF k = 0; k < Acol.rows; ++k)
		res += Acol(k) * conj(Brow(k));
	return alpha * res + beta * c;
}


template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf_transconja_notransb(TC c, const skepu::MatCol<TA> Acol, const skepu::MatCol<TB> Bcol, TS alpha, TS beta)
{
	TC res = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF k = 0; k < Acol.rows; ++k)
		res += conj(Acol(k)) * Bcol(k);
	return alpha * res + beta * c;
}

template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf_transconja_transb(TC c, const skepu::MatCol<TA> Acol, const skepu::MatRow<TB> Brow, TS alpha, TS beta)
{
	TC res = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF k = 0; k < Acol.rows; ++k)
		res += conj(Acol(k)) * Brow(k);
	return alpha * res + beta * c;
}

template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf_transconja_transconjb(TC c, const skepu::MatCol<TA> Acol, const skepu::MatRow<TB> Brow, TS alpha, TS beta)
{
	TC res = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF k = 0; k < Acol.rows; ++k)
		res += conj(Acol(k)) * conj(Brow(k));
	return alpha * res + beta * c;
}



template<typename TA, typename TB, typename TC>
void gemm(
	blas::Op                         transA,
	blas::Op                         transB,
	size_type                        m,
	size_type                        n,
	size_type                        k,
	scalar_type<TA, TB, TC>          alpha,
	Matrix<TA> BLAS_CONST&           A,
	size_type                        lda,
	Matrix<TB> BLAS_CONST&           B,
	size_type                        ldb,
	scalar_type<TA, TB, TC>          beta,
	Matrix<TC>&                      C,
	size_type                        ldc 
)
{
	typedef blas::scalar_type<TA, TB, TC> scalar_t;

  // constants
  const scalar_t zero = 0;
  const scalar_t one  = 1;

//  blas_error_if( lda < ((transA != Op::NoTrans) ? k : m) );
//  blas_error_if( ldb < ((transB != Op::NoTrans) ? n : k) );
//  blas_error_if( ldc < m );

  // quick return
  if (m == 0 || n == 0 || k == 0)
      return;
	
  if (alpha == zero)
	{
		auto skel_scale = Map<1>(gemm_uf_matrix_scale<TC, scalar_t>);
		skel_scale(C, C, beta);
    return;
  }

  // alpha != zero
  if (transA == Op::NoTrans)
	{
    if (transB == Op::NoTrans)
		{
			auto skel = skepu::Map<1>(gemm_uf_notransa_notransb<TA, TB, TC>);
			skel(C, C, A, B, alpha, beta);
    }
    else if (transB == Op::Trans)
		{
			auto skel = skepu::Map<1>(gemm_uf_notransa_transb<TA, TB, TC>);
			skel(C, C, A, B, alpha, beta);
    }
    else
		{ // transB == Op::ConjTrans
			auto skel = skepu::Map<1>(gemm_uf_notransa_transconjb<TA, TB, TC>);
			skel(C, C, A, B, alpha, beta);
    }
  }
  else if (transA == Op::Trans)
	{
    if (transB == Op::NoTrans)
		{
			auto skel = skepu::Map<1>(gemm_uf_transa_notransb<TA, TB, TC>);
			skel(C, C, A, B, alpha, beta);
    }
    else if (transB == Op::Trans)
		{
			auto skel = skepu::Map<1>(gemm_uf_transa_transb<TA, TB, TC>);
			skel(C, C, A, B, alpha, beta);
    }
    else
		{ // transB == Op::ConjTrans
			auto skel = skepu::Map<1>(gemm_uf_transa_transconjb<TA, TB, TC>);
			skel(C, C, A, B, alpha, beta);
    }
  }
  else
	{ // transA == Op::ConjTrans
    if (transB == Op::NoTrans)
		{
			auto skel = skepu::Map<1>(gemm_uf_transconja_notransb<TA, TB, TC>);
			skel(C, C, A, B, alpha, beta);
    }
    else if (transB == Op::Trans)
		{
			auto skel = skepu::Map<1>(gemm_uf_transconja_transb<TA, TB, TC>);
			skel(C, C, A, B, alpha, beta);
    }
    else
		{ // transB == Op::ConjTrans
			auto skel = skepu::Map<1>(gemm_uf_transconja_transconjb<TA, TB, TC>);
			skel(C, C, A, B, alpha, beta);
    }
  }
}
/*

// option A
map = Map<>(...);
map.setTriangular(skepu::Upper);
map(my_matrix);


O O O O 
O O O X
O O X X
O X X X

// option B
my_matrix = skepu::TriangleMatrix()

*/

#undef SKEPU_BLAS_STRIDE_TYPE_UF
#undef SKEPU_BLAS_SIZE_TYPE_UF
}}

static skepu::PrecompilerMarker endOfBlasHPP;
/* END BLAS.HPP INJECTION*/

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



struct skepu_userfunction_skepu_skel_1skel_gemv_uf_notrans_noconj
{
constexpr static size_t totalArity = 7;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
constexpr static bool usesPRNG = 0;
constexpr static size_t randomCount = SKEPU_NO_RANDOM;
using IndexType = void;
using ElwiseArgs = std::tuple<float>;
using ContainerArgs = std::tuple<skepu::MatRow<float>, skepu::Vec<float>>;
using UniformArgs = std::tuple<int, const float, const unsigned long, const float>;
typedef std::tuple<skepu::ProxyTag::MatRow, skepu::ProxyTag::Default> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
skepu::AccessMode::ReadWrite, skepu::AccessMode::ReadWrite, };

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float y, skepu::MatRow<float> Arow, skepu::Vec<float> x, int incx, const float alpha, const unsigned long m, const float beta)
{
	float tmp = 0;
	for (uint i = 0; i < m; ++i)
        tmp += x(i * incx) * Arow(i);
	return beta * y + alpha * tmp;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float y, skepu::MatRow<float> Arow, skepu::Vec<float> x, int incx, const float alpha, const unsigned long m, const float beta)
{
	float tmp = 0;
	for (uint i = 0; i < m; ++i)
        tmp += x(i * incx) * Arow(i);
	return beta * y + alpha * tmp;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "skepu_skel_1_my_thing_precompiled_MapKernel_gemv_uf_notrans_noconj.cu"
void thread_entry_gemv_cuda(skepu::Matrix<float>* A, skepu::Vector<float>* x, skepu::Vector<float>* y, size_t const m, float const split, float const alpha, float const beta) {
    skepu::blas::Op no_trans = skepu::blas::Op::NoTrans;
    skepu::backend::Map<1, skepu_userfunction_skepu_skel_1skel_gemv_uf_notrans_noconj, decltype(&skepu_skel_1_my_thing_precompiled_MapKernel_gemv_uf_notrans_noconj), void> skel(skepu_skel_1_my_thing_precompiled_MapKernel_gemv_uf_notrans_noconj);
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



struct skepu_userfunction_skepu_skel_0skel_gemm_uf_notransa_notransb
{
constexpr static size_t totalArity = 6;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 1;
constexpr static bool usesPRNG = 0;
constexpr static size_t randomCount = SKEPU_NO_RANDOM;
using IndexType = skepu::Index2D;
using ElwiseArgs = std::tuple<float>;
using ContainerArgs = std::tuple<const skepu::Mat<float>, const skepu::Mat<float>>;
using UniformArgs = std::tuple<float, float>;
typedef std::tuple<skepu::ProxyTag::Default, skepu::ProxyTag::Default> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
skepu::AccessMode::Read, skepu::AccessMode::Read, };

using Ret = float;

constexpr static bool prefersMatrix = 1;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(skepu::Index2D index, float c, const skepu::Mat<float> Arow, const skepu::Mat<float> Bcol, float alpha, float beta)
{
	float res = 0;
	for (uint k = 0; k < Arow.cols; ++k)
		res += Arow(index.row, k) * Bcol(k, index.col);
	return alpha * res + beta * c;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(skepu::Index2D index, float c, const skepu::Mat<float> Arow, const skepu::Mat<float> Bcol, float alpha, float beta)
{
	float res = 0;
	for (uint k = 0; k < Arow.cols; ++k)
		res += Arow(index.row, k) * Bcol(k, index.col);
	return alpha * res + beta * c;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "skepu_skel_0_my_thing_precompiled_MapKernel_gemm_uf_notransa_notransb.cu"
void thread_entry_gemm_cuda(skepu::Matrix<float>* A, skepu::Matrix<float>* B, skepu::Matrix<float>* C, size_t const m, size_t const n, double const split, float const alpha, float const beta) {
    // skepu::AccessMode readWriteMode = skepu::AccessMode::ReadWrite;
    // float alpha = 1.f;
    // float beta = 1.f;
// #ifndef SKEPU_NO_UPDATE_HOST
// #define SKEPU_NO_UPDATE_HOST
    skepu::backend::Map<1, skepu_userfunction_skepu_skel_0skel_gemm_uf_notransa_notransb, decltype(&skepu_skel_0_my_thing_precompiled_MapKernel_gemm_uf_notransa_notransb), void> skel(skepu_skel_0_my_thing_precompiled_MapKernel_gemm_uf_notransa_notransb);
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