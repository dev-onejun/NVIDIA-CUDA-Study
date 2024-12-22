---
two-columns: true
paper-title: "What Makes NVIDIA the Strongest: A Review of CUDA"
author: '
$$
\mathbf{\text{Wonjun Park}} \\
\text{Computer Science} \\
\text{University of Texas at Arlington} \\
\text{wxp7177@mavs.uta.edu}
$$
'
acronym_and_abbreviation: '
$$
\mathbf{\text{Acronym and Abbreviation}} \\
\begin{array}{|c|c|}
\hline
\text{Compute Unified Device Architecture (CUDA)} & \text{Central Processing Unit (CPU)} \\
\hline
\text{Graphics Processing Unit (GPU)} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$
'
---

### I. Introduction

### II. Literature Reviews

#### A. Background

**Terminologies** \
$\quad$

#### B. CUDA Kernel Execution

* GPU work is done in a thread
* Many threads run in parallel
* A collction of threads is a block. There are many blocks
* A collection of blocks is a grid.

* GPU functions are called kernels
* The execution configuration defines the number of blocks in a grid and the number of threads in block where every block in the grid has the same number of threads.

#### C. CUDA-Provided Thread Hierarchy Variables

* `gridDim.x`: the number of blocks in the grid
* `blockIdx.x`: the index of the current block in a grid
* `blockDim.x`: the number of threads in a block
* `threadIdx.x`: the index of the current thread in a block

**GPU-accelerated vs. CPU-only Applications** \
$\quad$ In CPU-only applications, data is sequentially processed by the CPU. This is a slow process, as the CPU can only process one task at a time. However, in GPU-accelerated applications, data is processed in parallel by the GPU. One of those that enables to process multiple tasks simultaneously is `cudaMallocManaged()` function which automatically migrates data to the GPU to perform a parallel computation. A task on the GPU is asynchronous so that the CPU can continue to process other tasks while the GPU is working.

Handling asynchronous tasks is an important and challenging. CUDA addresses this issue by `cudaDeviceSynchronize()` function which waits for all tasks on the GPU to complete before continuing. After synchronization, data which will be accessed by the CPU is also automatically migrated back to the CPU. With these concepts, GPU process multiple tasks simultaneously, which allows faster processing times.

**Hello Worlds** \
$\quad$

``` cuda
void CPUFunction()
{
    printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction()
{
    printf("This function is defined to run on the GPU.\n");
}

int main()
{
    CPUFunction();

    GPUFunction<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```
