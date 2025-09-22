---
layout: post
title:  "Performance Optimization of the ACT C++ Library on CPU and GPU"
date:   2025-09-21 23:11:27 -0700
categories: ACT
---

## Introduction

One of the goals of implementing the ACT C++ library is to be able to analyze EEG signals in real-time. 
In order to achieve the level of performance required we decided to explore different optimization strategies and to implement the library in a platform specific way.

## The Problem

The Matching Pursuit algorithm requires the generation of a potentially large dictionary of chirplets which is then greedily searched to find the best approximation atom for the signal.
The dictionary is a multi dimensional matrix residing in memory and the search operation involves multiplying the the signal with each atom in the dictionary to find the best match.
Parallel matrix multiplication is a well known problem and there are many ways to optimize it using BLAS, LAPACK, OpenMP, OpenCL, CUDA, etc.

The reference C++ implementation from the **ACT** class of the dictionary search is as follows:

{% highlight cpp %}
double ACT::inner_product(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

std::pair<int, double> ACT::search_dictionary(const std::vector<double>& signal) {
    assert(signal.size() == length);
    
    int best_idx = 0;
    double best_val = -std::numeric_limits<double>::infinity();
    
    for (int i = 0; i < dict_size; ++i) {
        double val = inner_product(dict_mat[i], signal);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    
    return std::make_pair(best_idx, best_val);
}
{% endhighlight %}

The dictionary search is a simple loop that iterates over the dictionary and finds the best match for the signal.

The dictionary search is called from the transform function n times, where n is the order of the transform. Each time the signal is updated to the residue which is the signal minus the best match found in the previous iteration.

{% highlight cpp %}

ACT::TransformResult ACT::transform(const std::vector<double>& signal, int order, double residual_threshold) {

    TransformResult result;
    result.params.resize(order, std::vector<double>(4));
    result.coeffs.resize(order);
    result.signal = signal;
    result.approx.resize(length, 0.0);
    result.residue = signal;  // Copy signal to residue

    double prev_resid_norm2 = std::numeric_limits<double>::max();
    
    for (int i = 0; i < order; ++i) {
        // Find best matching chirplet from dictionary
        auto [ind, val] = search_dictionary(result.residue);

        // Get coarse parameters
        std::vector<double> params = param_mat[ind];

        // Fine-tune parameters using BFGS optimization
        std::vector<double> refined_params = bfgs_optimize(params, result.residue);

        // Generate optimized chirplet
        auto updated_base_chirplet = g(refined_params[0], refined_params[1], refined_params[2], refined_params[3]);

        // Calculate coefficient (unit-energy atoms => LS amplitude is simple dot product)
        double updated_chirplet_coeff = inner_product(updated_base_chirplet, result.residue);

        // Store results for this order
        result.params[i] = refined_params;
        result.coeffs[i] = updated_chirplet_coeff;

        // Create new chirp component
        std::vector<double> new_chirp(length);
        for (int j = 0; j < length; ++j) {
            new_chirp[j] = updated_base_chirplet[j] * updated_chirplet_coeff;
        }

        // Update residue and approximation
        double resid_norm2 = 0.0;
        for (int j = 0; j < length; ++j) {
            result.residue[j] -= new_chirp[j];
            result.approx[j] += new_chirp[j];
            resid_norm2 += result.residue[j] * result.residue[j];
        }

        // Check for early stopping
        if (prev_resid_norm2 - resid_norm2 < residual_threshold) {
            //resize result.params and result.coeffs to i+1
            result.params.resize(i+1);
            result.coeffs.resize(i+1);
            break;
        }

        prev_resid_norm2 = resid_norm2;
    }
    

    // Calculate final error
    double resid_norm2 = 0.0;
    for (int j = 0; j < length; ++j) resid_norm2 += result.residue[j] * result.residue[j];
    result.error = std::sqrt(resid_norm2); // residual L2 norm
    
    return result;
}

{% endhighlight %}


## Optimization using BLAS

The dictionary search can be optimized using BLAS (Basic Linear Algebra Subprograms) GEMMV (**cblas_Xgemmv**) to perform the matrix multiplication in a highly optimized way, and BLAS IAMAX (**cblas_iXamax**) to find the maximum value in the vector.

ACT_CPU_T uses the Eigen library instead of the Standard Template Library (STL) to store the dictionary and the signal. In order to use BLAS we need to guarantee that matrix and vectors are stored contiguously in memory, which is the case for Eigen::Matrix and Eigen::Vector.

Here is the search_dictionary function as implemented in **ACT_CPU_T**:

{% highlight cpp %}

template <typename Scalar>
std::pair<int,Scalar> ACT_CPU_T<Scalar>::search_dictionary(const Eigen::Ref<const act::VecX<Scalar>>& signal) const {
    assert(signal.size() == length);
    if (dict_size == 0) return {0, 0.0};

    act::VecX<Scalar> scores(dict_size);
    scores.setZero();

    const int m = length;
    const int n = dict_size;
    const Scalar alpha = Scalar(1);
    const Scalar beta = Scalar(0);

    // scores = A^T * x  (A is m x n, column-major)
    act::blas::gemv_colmajor_trans(m, n,
                                   alpha,
                                   dict_mat.data(), m,
                                   signal.data(), 1,
                                   beta, scores.data(), 1);

    // Find argmax by magnitude using BLAS IAMAX (max |value|)
    int best_idx = act::blas::iamax(n, scores.data(), 1);
    Scalar best_val = scores[best_idx]; // keep signed value for potential diagnostics
    return {best_idx, best_val};
}
{% endhighlight %}

Note that we have templated the class to work with both double and single precision floating point numbers which is going to be required once we implement the GPU version of the library.

the act::blas::gemv_colmajor_trans and act::blas::iamax functions are wrapper for the correct BLAS function based on the template parameter :

{% highlight cpp %}
inline void gemv_colmajor_trans(int m, int n,
                                float alpha,
                                const float* A, int lda,
                                const float* x, int incx,
                                float beta,
                                float* y, int incy) {
    cblas_sgemv(CblasColMajor, CblasTrans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
inline void gemv_colmajor_trans(int m, int n,
                                double alpha,
                                const double* A, int lda,
                                const double* x, int incx,
                                double beta,
                                double* y, int incy) {
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

// iamax: index of the element with maximum absolute value
inline int iamax(int n, const float* x, int incx) {
    return static_cast<int>(cblas_isamax(n, x, incx));
}
inline int iamax(int n, const double* x, int incx) {
    return static_cast<int>(cblas_idamax(n, x, incx));
}
{% endhighlight %}


## Optimization using Apple Accelerate

When compiling on macOS, we can use the Apple Accelerate framework to take advantage of the vecLib library which provides highly optimized BLAS and LAPACK functions.
The Makefile adds the relevant `-framework Accelerate` flag to the compilation command.

ACT_Accelerate_T derives from ACT_CPU_T and overrides some core functions to specifically use the Accelerate framework (e.g. dot product) that we have used to test vDSP performance against BLAS, but it will be deprecated as we have found no benefits in using it.

## Optimization using Apple MLX

Apple MLX is a framework that provides a high level interface to the Apple Silicon GPU but also compiles using CUDA to run on NVIDIA GPUs.

The MLX version of search_dictionary is as follows:

{% highlight cpp %}
template <typename Scalar>
std::pair<int, Scalar> ACT_MLX_T<Scalar>::search_dictionary(const Eigen::Ref<const act::VecX<Scalar>>& signal) const {
    if constexpr (std::is_same_v<Scalar, float>) {
        // Lazily ensure MLX dictionary is available
        ensure_mlx_dict();

        const int m = this->get_length();
        const int n = this->get_dict_size();

        // Upload signal as float32 1D array directly from Eigen data (one host->device copy)
        mx::array x_arr(const_cast<float*>(signal.data()), mx::Shape{m}, mx::float32);

        // scores = A^T x, where A is (m x n) row-major on device
        // Compute via matmul(transpose(A), x)
        auto scores = mx::matmul(mx::transpose(*dict_gpu_), x_arr); // shape {n}

        // Argmax and best value
        auto idx_arr = mx::argmax(scores);
        int best_idx = idx_arr.template item<int>();
        auto best_val_arr = mx::take(scores, best_idx);
        float best_val_f = best_val_arr.template item<float>();
        return {best_idx, static_cast<Scalar>(best_val_f)};
    }
    // Fallback: CPU (Accelerate) path
    return Base::search_dictionary(signal);
}
{% endhighlight %}

The heavy lifting in this case is done by the mx:matmul function which expects a **row-major** matrix input, thus the need to ensure that the dictionary is transposed from the column-major layout used by Eigen. 
This is a one time operation that is done in the ensure_mlx_dict function.


## Profiling results

We used the `profile_act.cpp` program to profile the different versions of ACT.
The profile test generates a fairly large dictionary of approximately 600k atoms with a memory footprint of 2.3GB with double precision, 1.1GB with single precision.
The test then generates 5 synthetic signals that simulate EEG frequencies and a level of noise is added.
Each signal is then transformed using an order of 10, providing a good footprint for profiling performance.

We tested on two systems : Apple MacBook Pro M1 Max 32GB running MacOS 14.6 and a custom built Intel Core i9 14th gen, 64GB RAM, NVIDIA RTX 4090 running Ubuntu 24.04.

| System | Dict Search Mean (ms) | Dict Search Range (ms) | Transform Mean (ms) | Transform Range (ms) | Total Analysis Mean (ms) | Total Analysis Range (ms) | SNR Mean (dB) | SNR Range (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mac / ACT / Double Precision | 231.81 | [227.92, 244.91] | 2400.47 | [2317.07, 2569.64] | 2632.29 | [2546.01, 2797.56] | 13.26 | [13.05, 13.42] |
| Mac / CPU / Double Precision | 86.04 | [84.76, 89.51] | 894.08 | [877.65, 947.05] | 980.12 | [962.40, 1032.62] | 13.21 | [12.83, 13.77] |
| Mac / CPU / Single Precision | 52.43 | [50.45, 59.34] | 521.95 | [519.42, 523.65] | 574.38 | [569.87, 582.22] | 10.81 | [10.26, 11.36] |
| Mac / MLX / Single Precision | 4.12 | [3.95, 4.26] | 58.88 | [55.11, 61.79] | 63.01 | [59.06, 66.00] | 10.84 | [10.71, 10.99] |
| PC / MLX / Single Precision | 1.72 | [1.60, 1.96] | 35.32 | [33.21, 40.86] | 37.04 | [35.17, 42.55] | 10.74 | [9.90, 11.34] |
| PC / CPU / Single Precision | 153.33 | [151.69, 155.45] | 1544.03 | [1539.01, 1550.90] | 1697.36 | [1690.69, 1704.73] | 11.12 | [10.74, 11.29] |
