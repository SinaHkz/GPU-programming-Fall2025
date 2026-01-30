# Homework 4 — CUDA Programming
**Due Date:** January 2, 2025

This homework consists of **two CUDA programming problems** designed to strengthen your understanding of GPU computation, memory behavior, and performance analysis. Each problem focuses on a different but complementary aspect of CUDA programming: **parallel reduction and memory hierarchy**, and **image processing pipelines on the GPU**.

The emphasis of this homework is not only on correctness, but also on **performance evaluation, optimization, and reasoning about GPU execution behavior**.

You can upload your assignment using the following link: [Upload Form](https://forms.gle/iHP72xcaX1eab3sZ7)



## Problem 1 — Parallel Reduction and Memory Behavior

In this problem, you will implement a **parallel reduction (sum of an array)** using CUDA. You will explore how different memory allocation and access strategies affect performance on the GPU.

You are expected to:

* Implement reduction kernels using different memory models
* Compare global memory–based and shared memory–based reductions
* Analyze memory access efficiency and performance
* Optimize the reduction to improve execution time

This problem aims to build a solid understanding of **CUDA memory hierarchy**, **shared memory usage**, and **block-level parallelism**.



## Problem 2 — Image Processing Pipeline Using CUDA

In this problem, you will implement a **CUDA-based image processing pipeline** consisting of multiple kernels executed in sequence.

You will:

* Convert RGB images to grayscale using a CUDA kernel
* Apply Sobel edge detection on the grayscale images
* Process multiple images efficiently on the GPU
* Compare GPU performance with a serial CPU implementation

The goal of this problem is to expose you to **realistic multi-kernel CUDA workflows**, shared memory optimization for stencil-based computation, and end-to-end performance evaluation.


## Final Submission Checklist

* Code for all 2 problems
* Simple documentation for each problem
* Required images and profiling tables

Keep the write‑ups short—just explain what you did and show your results.

## Academic Integrity
All work must be your own. Profiling results must be collected on your own machine. Cite any references you use.
