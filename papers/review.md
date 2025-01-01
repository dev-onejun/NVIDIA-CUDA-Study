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
$\quad$ With CUDA programming, GPU work is performed by **threads**, which run in parallel. These threads are grouped into **blocks**, and multiple blocks form a **grid**. The functions executed on the GPU are known as kernels. The executino configuration specifies **the number of blocks in a grid** and **the number of threads in each block**, where every block in the grid containing the same number of threads. These relationships are called **thread hierarchy**.

To manage the thread hierarchy, CUDA provides several built-in variables. `gridDim.x` represents the total number of blocks in the grid. `blockIdx.x` indicates the index of the current block within the grid. `blockDim.x` denotes the number of threads in each block, and `threadIdx.x` specifies the index of the current thread within its block.

**GPU-accelerated vs. CPU-only Applications** \
$\quad$ In CPU-only applications, data is sequentially processed by the CPU. This is a slow process, as the CPU can only process one task at a time. However, in GPU-accelerated applications, data is processed in parallel by the GPU. One of those that enables to process multiple tasks simultaneously is `cudaMallocManaged()` function which automatically migrates data to the GPU to perform a parallel computation. A task on the GPU is asynchronous so that the CPU can continue to process other tasks while the GPU is working.

Handling asynchronous tasks is an important and challenging. CUDA addresses this issue by `cudaDeviceSynchronize()` function which waits for all tasks on the GPU to complete before continuing. After synchronization, data which will be accessed by the CPU is also automatically migrated back to the CPU. With these concepts, GPU process multiple tasks simultaneously, which allows faster processing times.

**Hello Worlds**

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

$\quad$ The above code demonstrates a simple example of a CPU function and a GPU function. While Code executed on the CPU is referred to as **host code**, code running on the GPU is called **device code**. The `__global__` keyword indicates that the following function can run on either the CPU or the GPU. Importantly, the return type of the function must be `void`.

When it comes to a line of calling function `GPUFunction<<<1, 1>>>();`, the function is called as a **kernel**. The kernel necessarily requires an **execution configuration** in the form of `<<<gridDim (the number of blocks), blockDim (the number of threads)>>>`.

As the paper mentioned earlier, the kernel is executed on the GPU, meaning that it runs asynchronously, unlike most C/C++ code; thus, the rest of the code without the kernel will continue to execute without waiting for the kernel to complete. In order to synchronize this gap between the CPU and the GPU, `cudaDeviceSynchronize()` function causes the host code to wait until the device code completes, and only then resumes execution on the CPU.



* The maximum number of threads per block is 1024. -> Using blocks is inevitable to handle more threads.
    - threadIdx.x + blockIdx.x * blockDim.x = dataIndex (map each thread to a unique element in the vector)
    - Note that although the executing sequence of the threads is guaranteed, the executing sequence of the blocks is not guaranteed.

* Memory Management

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

    - malloc() variable is not able to use on the GPU, while cudaMallocManaged() is able to use on both the CPU and the GPU.









### References

$$\tag*{1}\label{[1]} \text{[1] }$$
