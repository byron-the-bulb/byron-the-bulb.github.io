---
layout: post
title:  "Introduction to Adaptive Chirplet Transformation using the ACT C++ Library"
date:   2025-09-18 14:38:29 -0700
categories: ACT
---
[[Github : ACT C++ Library]](https://github.com/nemes-inc/ACT-lib)

Adaptive Chirplet Transform (ACT) is a signal processing technique that extends the concept of chirplet transform (CT) to adaptively select the best chirplet basis functions for representing a given signal.

We will be using ACT to analyze signals in the time-frequency domain and extract features that can be used to classify signals, with a focus on EEG signals.

We will also be exploring various optimization mathods to improve the performance of ACT, with the goal of analyzing multiple EEG signals in real-time.

This post includes an interactive workbench powered by p5.js to experience the results of the ACT analysis on EEG signals via visual and audio feedback. 
I am a sensory learner and I found it very helpful to have a visual and audio representation of the results. 

This work is based and inspired by the research papers and the code referenced at the end of this post.

## The ACT algorithm

The ACT algorithm used here implements the two step matching pursuit followed by BFGS optimization outlined in the cited papers and the code referenced at the end of this post.

For a signal of a given sample rate and length, the following steps are performed:

1. Generate a dictionary of chirplets (atoms) from four input parameters: time center (tc), frequency center (fc), time duration (logDt), and chirp rate (c). Each input parameter is specified as a min, max and step values which define the multi-dimensional matrix/grid that is the dictionary, i.e. the search space. 
The dictionary is generated once and reused for the rest of the analysis.
2. Use Matching Pursuit (MP) to find the best approximation atom for the signal. The current implementation is a simple greedy MP algorithm that selects the best atom at each iteration. The simplicify of the implementation using basic matrix and vector operations is that it is easy to parallelize and can be implemented in a GPU.
3. Refine the selected atom's parameters via BFGS over (tc, fc, logDt, c) to maximize correlation. Estimate the optimal coefficient (least-squares against unit-energy template).
4. Subtract the reconstructed chirplet from the residual and repeat steps (2–4) up to the chosen transform order K or until the residual energy is below the residual threshold.
5. The result of the process is a set of chirplets that, when combined, best represent the signal using the provided dictionary, i.e. the reconstructed signal, and a residual signal that is the difference between the original signal and the reconstructed signal which is everything else that is not represented by the dictionary (noise, artifacts, etc.).
A chirplet coefficient is also returned which represents the energy of the chirplet in the signal.

The signal length is fixed at the time of dictionary generation and cannot be changed. The signal length is specified in samples and the sample rate is specified in Hz.

## Defining the dictionary

The dictionary is defined by the following parameters:

- `tc_min`, `tc_max`, `tc_step`: Time center (in samples)
- `fc_min`, `fc_max`, `fc_step`: Frequency center (Hz)
- `logDt_min`, `logDt_max`, `logDt_step`: Log duration
- `c_min`, `c_max`, `c_step`: Chirp rate (Hz/s)

**Time center** is the center of the chirplet within the length of the signal, i.e. the sample at which the chirplet is centered. Time center is bounded by signal length.

**Frequency center** is the center of the chirplet in frequency, i.e. the frequency at which the chirplet is centered.

**Log duration** is the duration of the chirplet in log space, i.e. the duration of the chirplet in ms. 

**Chirp rate** is the rate at which the chirplet chirps, i.e. the rate at which the chirplet changes frequency.

## The ACT C++ Library

The basic interface of the ACT C++ library consist of three main functions:

{% highlight cpp %}

// Initialize the ACT object
ACT act(double sample_rate, int signal_length, const ACT::ParameterRanges& ranges);

// Generate the chirplet dictionary
act.generate_chirplet_dictionary();

// Transform the signal
ACT::TransformResult res = act.transform(double* signal, int order, double residual_threshold);

{% endhighlight %}

The transform function returns a TransformResult object that contains the following information:

- `chirplets`: A vector of Chirplet objects that represent the best approximation atoms for the signal.
- `residual`: A vector of double values that represent the residual signal after the transform.
- `coeffs`: A vector of double values that represent the optimal coefficients for the chirplets.

The **ParameterRanges** struct is defined as follows:

{% highlight cpp %}
struct ParameterRanges {
      double tc_min, tc_max, tc_step;      // Time center (in samples)
      double fc_min, fc_max, fc_step;      // Frequency center (Hz)
      double logDt_min, logDt_max, logDt_step;  // Log duration
      double c_min, c_max, c_step;         // Chirp rate (Hz/s)
};
{% endhighlight %}

The **TransformResult** struct is defined as follows:

{% highlight cpp %}
struct TransformResult {
    std::vector<std::vector<double>> params;  // [order x 4] chirplet parameter matrix
    std::vector<double> coeffs;               // chirplet coefficients
    std::vector<double> signal;               // original signal
    double error;                             // residual error
    std::vector<double> residue;              // final residue
    std::vector<double> approx;               // approximation
};
{% endhighlight %}

## Platform specific implementation

  The ACT C++ library is implemented with a focus on performance optimization across different compute platforms. 

  At the moment we have the following platform specific implementations:
 - `class ACT`: Reference implementation using standard library with double precision.
 - `class ACT_CPU_T<Scalar>`: Base implementation using Eigen and BLAS GEMM for matrix operations, compiles on Intel and Mac. Templatized to support single and double precision.
 - `class ACT_Accelerate_T : public ACT_CPU_T<Scalar>`: Implementation using Accelerate framework (vDSP) for Apple Silicon.
 - `class ACT_MLX_T : public ACT_Accelerate_T<Scalar>`: Implementation using Apple MLX library for GPU acceleration. Targets Apple Metal and Nvidia CUDA.

  BFGS is implemented using ALGLIB `alglib::minbcoptimize` on all platforms.

  We will cover the specific implementation details of each platform in a separate post.


## Interactive ACT Workbench (p5.js)
Below is an embedded interactive chirplet viewer powered by p5.js. 
The workbench includes the results of a few analysis of EEG signals using the ACT C++ library.
The EEG data used is from the recent Muse Sleep Study dataset.

Pick an analysis from the dropdown menu to visualize and playback the results.
An interesting observation is to compare the original signal with the reconstruction. I have included different order of analysis to compare the quality of reconstruction based on order.
The dictionary used is focused on the sleep study frequencies of interest, i.e. sub 20 Hz.

{% include act_p5_embed.html %}


## Additional media

[Loom recording of me using the ACT workbench](https://www.loom.com/share/dd5a483e554a4fe3b87c603fe1dfc135?sid=e20dd7ec-2b79-4167-a0fe-d998be419de9)

## References:

[https://github.com/jiecui/mpact](https://github.com/jiecui/mpact)

[https://github.com/amanb2000/Adaptive_Chirplet_Transform](https://github.com/amanb2000/Adaptive_Chirplet_Transform)


- *Steve Mann and Simon Haykin "Adaptive chirplet: an adaptive generalized wavelet-like transform", Proc. SPIE 1565, Adaptive Signal Processing, (1 December 1991); https://doi.org/10.1117/12.49794*
- *J. Cui, W. Wong and S. Mann, "Time-frequency analysis of visual evoked potentials by means of matching pursuit with chirplet atoms," The 26th Annual International Conference of the IEEE Engineering in Medicine and Biology Society, San Francisco, CA, USA, 2004, pp. 267-270, doi: 10.1109/IEMBS.2004.1403143.*