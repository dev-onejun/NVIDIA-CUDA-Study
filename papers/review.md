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
\text{Graphics Processing Unit (GPU)} & \text{Deep Learning Institute (DLI)} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$
'
---

### I. Introduction

$\quad$ As a part of NVIDIA Developer Program, the NVIDIA Deep Learning Institute (DLI) [[1](#mjx-eqn-1)] offers a free course among their self-paced course] offers a free course among their self-paced course] offers a free course among their self-paced course] offers a free course among their self-paced courses.

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

**Parallel Programming Configuration** \
$\quad$ Parallel computing is a type of computation in which many calculations or processes are carried out simultaneously. Each data element is processed by each thread. However, the maximum number of threads per block that CUDA defines is so finite, 1024, that it is inevitable to use blocks to concurrently handle more threads. An **dataIndex** represents the index of the thread corresponding to the index of the data element in a grid. Calculated by $\text{threadIdx.x} + \text{blockIdx.x} \times \text{blockDim.x}$, the `dataIndex` enables to access all threads in the grid called by a single kernel.

Three possible cases are prompted regarding the relationship between the number of threads $T$ and the number of data elements $N$; **1. $\mathbf{T \eq N}$** Nothing needs to be considered in this case. **2. \mathbf{T \gt N}$** This makes empty threads, which are not used, so that handling the case as checking whether `dataIndex` is smaller than $N$ is necessary. **3. \mathbf{T \lt N}$** This case requires a **grid-stride loop** technique. The technique allows a single thread to stride forward sequentially among the data elements by the number of threads in the grid with $\text{blockDim.x} \times \text{gridDim.x}$. The `dataIndex` in this technique is consequently calculated by $(\text{threadIdx.x} + (\text{blockIdx.x} \times \text{blockDim.x}) + (\text{blockDim.x} \times \text{gridDim.x}) \times i)$ where $i$ is the iteration index.

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

**a** \
$\quad$






### References

$$\tag*{}\label{1} \text{[1] NVIDIA Deep Learning Institute, https://learn.nvidia.com/en-us/training/self-paced-courses, accessed in Jan. 3rd, 2025}$$
