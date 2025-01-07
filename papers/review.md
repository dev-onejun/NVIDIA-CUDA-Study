---
two-columns: true
paper-title: "What Makes NVIDIA the Strongest: A Review of CUDA"
author: '
$$
\begin{array}{c}
\hline
\\
\text{A CUDA Overview: Basics for NVIDIA GPU-accelerated Programming} \\
\\
\hline
\end{array} \\
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
\text{Graphics Processing Unit (GPU)} & \text{Deep Learning Institute (DLI)} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$
'
---

### I. Introduction

$\quad$ As a part of NVIDIA Developer Program, the NVIDIA Deep Learning Institute (DLI) [[1](#mjx-eqn-1)] offered a free course among their self-paced course.

### II. Literature Reviews

#### A. Background

**Terminologies** \
$\quad$ With CUDA programming, GPU work is performed by **threads**, which run in parallel. These threads are grouped into **blocks**, and multiple blocks form a **grid**. The functions executed on the GPU are known as kernels. The executino configuration specifies **the number of blocks in a grid** and **the number of threads in each block**, where every block in the grid containing the same number of threads. These relationships are called **thread hierarchy**.

To manage the thread hierarchy, CUDA provides several built-in variables. `gridDim.x` represents the total number of blocks in the grid. `blockIdx.x` indicates the index of the current block within the grid. `blockDim.x` denotes the number of threads in each block, and `threadIdx.x` specifies the index of the current thread within its block.

**GPU-accelerated vs. CPU-only Applications** \
$\quad$ In CPU-only applications, data is sequentially processed by the CPU. This is a slow process, as the CPU can only process one task at a time. However, in GPU-accelerated applications, data is processed in parallel by the GPU. One of those that enables to process multiple tasks simultaneously is `cudaMallocManaged()` function which automatically migrates data to the GPU to perform a parallel computation. A task on the GPU is asynchronous so that the CPU can continue to process other tasks while the GPU is working.

Handling asynchronous tasks is an important and challenging. CUDA addresses this issue by `cudaDeviceSynchronize()` function which waits for all tasks on the GPU to complete before continuing. After synchronization, data which will be accessed by the CPU is also automatically migrated back to the CPU. With these concepts, GPU process multiple tasks simultaneously, which allows faster processing times.

#### B. Hello Worlds

**Hello World in CUDA**

``` cuda
void CPUFunction() {
    printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction() {
    printf("This function is defined to run on the GPU.\n");
}

int main() {
    CPUFunction();

    GPUFunction<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```

$\quad$ The above code demonstrates a simple example of a CPU function and a GPU function. While Code executed on the CPU is referred to as **host code**, code running on the GPU is called **device code**. The `__global__` keyword indicates that the following function can run on either the CPU or the GPU. Importantly, the return type of the function must be `void`.

When it comes to a line of calling function `GPUFunction<<<1, 1>>>();`, the function is called as a **kernel**. The kernel necessarily requires an **execution configuration** in the form of `<<<gridDim (the number of blocks), blockDim (the number of threads)>>>`.

As the paper mentioned earlier, the kernel is executed on the GPU, meaning that it runs asynchronously, unlike most C/C++ code; thus, the rest of the code without the kernel will continue to execute without waiting for the kernel to complete. In order to synchronize this gap between the CPU and the GPU, `cudaDeviceSynchronize()` function causes the host code to wait until the device code completes, and only then resumes execution on the CPU.

**Parallel Programming Configuration** \
$\quad$ Parallel computing is a type of computation in which many calculations or processes are carried out simultaneously. Each data element is processed by each thread. However, the maximum number of threads per block that CUDA defines is so finite, 1024, that it is inevitable to use blocks to concurrently handle more threads. Furthermore, traits of GPU hardware often make the desirable number of threads per block to be a multiple of 32 due to getting the performance benefits.

An **dataIndex** represents the index of the thread corresponding to the index of the data element in a grid. Calculated by $\text{threadIdx.x} + \text{blockIdx.x} \times \text{blockDim.x}$, the `dataIndex` enables to access all threads in the grid called by a single kernel.

Three possible cases are prompted by the hardware requirements regarding the relationship between the number of threads $T$ and the number of data elements $N$; **1. $\mathbf{T = N}$** Nothing needs to be considered in this case. **2. $\mathbf{T \gt N}$** This makes empty threads, which are not used, so that handling the case as checking whether `dataIndex` is smaller than $N$ is necessary.

**3. $\mathbf{T \lt N}$** This case requires a **grid-stride loop** technique. The technique allows a single thread to stride forward sequentially among the data elements by the number of threads in the grid with $\text{blockDim.x} \times \text{gridDim.x}$. The `dataIndex` in this technique is consequently calculated by $\text{threadIdx.x} + (\text{blockIdx.x} \times \text{blockDim.x}) + (\text{blockDim.x} \times \text{gridDim.x}) \times i$ where $i$ is the iteration index. The following code block shows the implementation of the grid-stride loop technique.

``` cuda
__global__ void kernel(int *a, int N) {
    int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = blockDim.x * gridDim.x;

    for (int i = indexWithinTheGrid; i < N; i += gridStride) {
        // do work on a[i];
    }
}
```

$\quad$ When both the number of threads and the number of data elements were known while the number of blocks was not, the following code ensures that the extra block is created to handle all data elements.

``` cuda
int N = 100000;
size_t threads_per_block = 256;

// calculate the number of blocks from given variables
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

some_kernel<<<number_of_blocks, threads_per_block>>>(N);

__global__ void some_kernel(int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // do something
    }
}
```

Note that the executing sequence of the threads is guaranteed, but the executing sequence of the blocks is not guaranteed.

**Memory Management** \
$\quad$ The following code shows how to allocate and free memory on the GPU.

``` cuda
// CPU-only
int N = 2 << 20; // 2^21
size_t size = N * sizeof(int);

int *a;
a = (int *)malloc(size);

free(a);

// GPU-accelerated
int N = 2 << 20;
size_t size = N * sizeof(int);

int *a;
cudaMallocManaged(&a, size);

cudaFree(a);
```

Note that a memory address assigned by `malloc()` is not able to access on the GPU, while a memory address assigned by `cudaMallocManaged()` is able to access on both the CPU and the GPU.

**Error Handling** \
$\quad$ Error handling is crucial in all programming languages. All errors are typed by `cudaError_t`. Three types of principal errors is addressed in this paper; **1. From Memory Allocation** Due to hardware limitations, memory allocation often fail. A CUDA library function, `cudaMallocManaged()`, returns a `cudaError_t` type, which can be used to check whether the memory allocation was successful. The following code snippet shows how to handle memory allocation errors.

``` cuda
cudaError_t err;
err = cudaMallocManaged(&a, size);  // Assume the existence of `a` and `N`

if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString()` is provided by CUDA.
}
```

**2. During Launching a Kernel** As mentioned above, the kernel should be defined by void return type which prohibits from handling errors in the kernel. In order to unfold this issue, CUDA gives a function `cudaGetLastError()` which returns the last error as a type of `cudaError_t`. An error evoked by the kernel launching is examined by the following example.

``` cuda
someKernel<<<1, -1>>>();  // the parameters of kernels cannot be negatibve

cudaError_t err;
err = cudaGetLastError();

if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

**3. During Executing a Kernel** The previous two cases account for the errors in synchronous code flow. However, it is not underestimated that the asynchronous execution in GPUs is mainly a regard of parallel programming. The `cudaDeviceSynchronize()` function returns an error which emerges during the execution of the kernel.

``` cuda
cudaError_t err;
err = cudaDeviceSynchronize();

if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

These errors can be wrapped up by a macro function like the following example.

``` cuda
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

int main() {
    /*
     * The macro above can be wrapped around any function
     * that returns a value of type `cudaError_t`.
     */
    checkCuda(cudaMallocManaged(&a, size));
}
```






### References

Most code snippets are provided by the course material.

$$
\tag*{}\label{1} \text{[1] NVIDIA Deep Learning Institute, https://learn.nvidia.com/en-us/training/self-paced-courses, accessed in Jan. 3rd, 2025}
$$
