---
layout: post
title:  "Multi Signals Performance Optimization of the ACT C++ Library on CPU and GPU"
date:   2025-10-01 10:11:27 -0700
categories: ACT
---

[[Github : ACT C++ Library]](https://github.com/nemes-inc/ACT-lib)

In the previous post [Performance Optimization of the ACT C++ Library on CPU and GPU](/ACT-performance) we covered hardware acceleration and performance optimization to transform a single signal using the ACT algorithm.
Here we will cover how to transform multiple signals in parallel in real time!


## Handling multiple signals

As the performance profiling we did in the previous post showed, the bulk of the computation in the ACT algorithm is the dictionary search, which is a matrix multiplication between the dictionary matrix and the signal vector, all other operations, such as the BFGS optimization, are minor in comparison.

Given that multiple signals vectors can be represented as a matrix, we can perform the dictionary search in parallel by simply multiplying the dictionary matrix by the matrix of signals.

**BLAS** offers the `gemm` function to perform matrix multiplication for our CPU only implementation, and **MLX** offers the `mx::matmul` function to do the same on GPUs.

Here are the relevant code snippets for the CPU and GPU implementations:

### CPU with BLAS

{% highlight cpp %}
    for (int it = 0; it < opts.order; ++it) {

        // Coarse scores for all signals at once: S = A^T * R
        act::blas::gemm_colmajor(CblasTrans, CblasNoTrans,
                                  n, k, m,
                                  Scalar(1),
                                  A.data(), m,
                                  R.data(), m,
                                  Scalar(0),
                                  S.data(), n);

        for (int j = 0; j < k; ++j) {

            const Scalar* Sj = S.data() + static_cast<size_t>(j) * static_cast<size_t>(n);
            int best_idx = act::blas::iamax(n, Sj, 1);

            // Refinement if requested
            Eigen::Vector4d refined = init;
            if (opts.refine) {
                refined = act.refine_params_bfgs(init, R.col(j));
            }           
 {% endhighlight %}

All signals are sequentially processed at the same order `opts.order` in the main `for` loop.
For each order `it`, we multiply the dictionary matrix `A` by the signal matrix `R` to get the scores matrix `S`.

Then we find the best match for each signal in the scores matrix `S` using the `act::blas::iamax` function.
The best match is used as an initial guess for the BFGS optimization.


### GPU with MLX

{% highlight cpp %}
        std::vector<float> R_rowmajor(static_cast<size_t>(m) * static_cast<size_t>(k));

        for (int it = 0; it < opts.order; ++it) {
            if (active_count == 0) break;

            // Upload R
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < k; ++j) {
                    R_rowmajor[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)] = R(i, j);
                }
            }
            mx::array R_gpu(R_rowmajor.data(), mx::Shape{m, k}, mx::float32);

            // Coarse selection on device
            auto S = mx::matmul(mx::transpose(A_gpu), R_gpu); // {n, k}
            auto idxs = mx::argmax(mx::abs(S), 0);            // {k}

            for (int j = 0; j < k; ++j) {

                int best_idx = mx::take(idxs, j).template item<int>();

                // Coarse init
                Eigen::Vector4d init;
                init[0] = P(best_idx, 0);
                init[1] = P(best_idx, 1);
                init[2] = P(best_idx, 2);
                init[3] = P(best_idx, 3);

                // Refine in double (public wrapper handles internal double optimize)
                Eigen::Vector4d refined = init;
                if (opts.refine) {
                    // Map current float residual to Eigen::Map<float> and pass; wrapper accepts Scalar=Scalar(act)
                    refined = act.refine_params_bfgs(init, R.col(j));
                }
{% endhighlight %}

Similar to the CPU implementation we process all of the signals at the same order `opts.order` in the main `for` loop.

The signal matrix `R` is uploaded to the GPU, and `mx::matmul` is used to perform the matrix multiplication between the transposed dictionary matrix `A_gpu` and the signal matrix `R_gpu` to get the scores matrix `S`.

For each signal we then use `mx::take` to find the best match for the signal in the scores matrix `S`.
The best match is used as an initial guess for the BFGS optimization.

## Profiling results

The profiler for the multi-signal is configured similarly to the single signal profiler, the code is in `profile-act-mt.cpp`.
The code generates a fairly large dictionary of approximately 600k atoms with a memory footprint of 2.3GB with double precision, 1.1GB with single precision.
The signal length is 2 seconds, 512 samples at 256Hz.
By default 16 separate signals are generated and transformed in parallel, the performance numbers collected below use this configuration.
The transform order for all signals is 10.

- **Mac** : Apple MacBook Pro M1 Max 32GB running MacOS 14.6 
- **PC** : custom built Intel Core i9 14th gen, 64GB RAM, NVIDIA RTX 4090 running Ubuntu 24.04.

 - **CPU** : the ACT implementation using BLAS
 - **MLX** : the ACT implementation using Apple MLX

  - **double** : double precision (`double|double64`)
  - **single** : single precision (`float|float32`)

 
 - **Transform** : the full transform operation at order 10, this includes the dictionary search operation and the BFGS optimization at each turn
 - **SNR** : Signal to Noise Ratio, higher is better
 - **RT Factor** : Real Time Factor, higher is better, this is the ratio of the transform time to the signal length in seconds, in this specific case how many 2 seconds of EEG data can be processed in one second, which will give us a confidence factor on the real time performance of the algorithm.
 
| Platform | Transform | SNR | RT Factor |
| --- | --- | --- | --- |
| Mac / CPU / double | Total batch: 1123.43 ms; Per-signal avg: 70.21 ms | Mean: 13.10 dB; Range: [12.74, 13.77] dB | 28.5x |
| Mac / MLX / single | Total batch: 423.13 ms; Per-signal avg: 26.45 ms | Mean: 10.96 dB; Range: [10.56, 11.69] dB | 75.6x |
| PC / MLX / single | Total batch: 208.20 ms; Per-signal avg: 13.01 ms | Mean: 11.15 dB; Range: [10.26, 12.20] dB | 153.7x |


The results clearly show that the ACT library is capable of processing 16 signals at a pace much faster than real time on all platforms.


## Demo

To showcase the realtime capabilities I added an OSC receiver and parser to the ACT EEG workbench application. 
The OSC receiver is configured to receive data from the Muse headset and parse it into a matrix of signals, which are then transformed in real time using the ACT algorithm and played back as audio waveforms to the platform sound system.
The current implementation uses the raw signal from each of the 4 EEG channels of the Muse headset without any filtering or preprocessing.

Here is a video of my brainwave signals played back as chirplets:
[Loom](https://www.loom.com/share/b28944d4ead94ae2833c5a74f254b096?sid=cd1203f4-d32d-4701-a5a7-9e6b6439672b)



 