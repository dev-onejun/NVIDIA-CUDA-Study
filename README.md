# NVIDIA CUDA Study

https://learn.nvidia.com/en-us/training/self-paced-courses

## Compile and Run CUDA Programs

``` bash
$ nvcc -o hello-gpu 01-hello/01-hello-gpu.cu -run
```

* `-o` option specifies the output file name. (same as gcc or clang)
* With `-run` option, the compiled program will be executed immediately after the compilation command.
