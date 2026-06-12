# Berry-Suite-Versions

This repository contains multiple versions of several core scripts from the Berry Suite of programs, developed to investigate and benchmark approaches for reducing storage requirements and execution time. The repository serves as a collection of optimization implementations, experimental modifications, and performance-oriented variants of existing Berry Suite workflows.

The primary goal of this repository is to provide a framework for systematically exploring optimization strategies within the Berry Suite ecosystem. The collection allows direct comparison between alternative implementations while monitoring their effects on:

- Computational performance
- Storage footprint
- Numerical accuracy
- Physical consistency of the results

This repository contains experimental code and optimization prototypes. Some implementations may be incomplete, exploratory, or intended solely for benchmarking purposes.

## Included Scripts

Different optimized (or optimization-attempt) versions of the following Berry Suite components are provided:

- `generatewfc.py`
- `r2k.py`
- `shg.py`
- `dotproduct.py`
- `basisrotation.py`
- `berry_geometry.py`

These versions explore techniques aimed at:

- Reducing intermediate and output file sizes
- Accelerating computationally intensive operations
- Evaluating trade-offs between compression, accuracy, and performance
- Testing alternative data representations and processing strategies

## Helper Utilities

The repository also includes four helper modules designed to support performance and compression studies:

#### 1. Entropy Analysis

Utilities for calculating data entropy, enabling estimation of the theoretical compressibility of generated datasets.

#### 2. Accuracy and Continuity Metrics

Functions for evaluating the impact of optimizations through:

- L2 error calculations
- Wavefunction overlap error analysis
- Band continuity metrics

These tools help quantify deviations introduced by compression or optimization techniques.

#### 3–4. Compression Testing Utilities

Experimental helper modules for testing and benchmarking different data compression methods, including:

- Compression ratio evaluation
- Reconstruction accuracy assessment
- Compression/decompression performance measurements
