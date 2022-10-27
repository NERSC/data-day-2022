# Intro to Parallel Python at NERSC

This guide introduces the basics of how to write parallel Python programs at NERSC.

This guide assumes you are new to Python and/or parallel programming on an HPC system.

It may be helpful to review a [Python tutorial](https://docs.python.org/3/tutorial/), the [numpy quickstart guide](https://numpy.org/doc/stable/user/quickstart.html), or the [numpy beginner's guide](https://numpy.org/doc/stable/user/absolute_beginners.html).

To learn the basics of job submission at NERSC using Slurm, please see the [running jobs](https://docs.nersc.gov/jobs/) page in the NERSC documentation.

This guide will begin with a brief background of parallelism in Python and walk through an example of common indirect parallelism in Python.

Next we will walk through a few techniques for solving embarissingly parallel problems with direct parallel programming in Python.
We'll consider solutions that leverage the following:

 * serial `for`-loop
 * multiprocessing
 * mpi4py
 * dask
 * cupy

This guide is not exhaustive and should be considered as a starting point for parallel Python programming at NERSC.

Please see the [Python at NERSC](https://docs.nersc.gov/development/languages/python/) page for more information about Python at NERSC.

## Getting started

Log in to NERSC (this tutorial assumes you will be using Perlmutter).
Download or clone this repository using git into your `$SCRATCH` directory and navigate to this directory.

```bash
# log in to NERSC (Perlmutter)
# change directory to scratch
cd $SCRATCH
# clone the examples repository
git clone https://gitlab.com/NERSC/python-examples.git
# change directory to the parallel-python python examples
cd python-examples/parallel-python
```

## Parallelism in Python

### The GIL

It's hard to discuss parallelism in Python without mentioning the [Global Interpreter Lock](https://docs.python.org/3/glossary.html#term-global-interpreter-lock) (GIL, pronounced "gill").
In short, the GIL simplifies the implementation of the Python interpreter by limiting what we can do with [threads](https://en.wikipedia.org/wiki/Thread_(computing)) in Python.
The examples in this guide work around the limitations imposed by the GIL.

### Indirect Parallelism in Python

If you have experience using NumPy or SciPy, then you already have experience with parallel Python programming.

Consider the following code which computes an eigenvalue decomposition of a random symmetric positive definite matrix:

```python
import numpy as np

# construct a random symmetric positive definite matrix
n = 1000
b = np.random.rand(n, n)
a = b.T @ b

# compute eigenvalue decomposition
w, v = np.linalg.eigh(a)
```

It may not seem like it but that is an example of Python code that is capable of leveraging multi-core systems.
Under the hood, many [linear algebra methods in NumPy use a BLAS backend](https://numpy.org/install/#numpy-packages--accelerated-linear-algebra-libraries) such as OpenBLAS or Intel's MKL, which may use multiple threads.
The multithreading parallelism in lower level backends used by NumPy is not constrained by Python's GIL.

Let's use Python's `timeit` module to benchmark the eigenvalue decomposition step on a Perlmutter CPU node.
The [`timeit`](https://docs.python.org/3/library/timeit.html) module is helpful for measuring the execution time of small snippets of code while avoiding common benchmarking pitfalls.

To begin, start an interactive job on a CPU node and load the python module.

```bash
> salloc -C cpu -A <account> -N 1 -q interactive -t 60
> module load python
```

We'll use `timeit`'s `-s` option to specify the setup statements followed by the statement to benchmark.
`timeit` will execute the setup once and then execute the benchmark statement several times to obtain a measurement.

```bash
> python -m timeit -s "import numpy as np; n = 1000; b = np.random.rand(n, n); a = b.T @ b" "np.linalg.eigh(a)"
1 loop, best of 5: 427 msec per loop
```

Exercise: Try adjusting the value of `n` in the previous command. How does the performance vary with respect to the size of the matrix?

<!--- Answer
```bash
> for n in 10 20 50 100 200 500 1000 2000; do echo -n "n=$n | "; python -m timeit -s "import numpy as np; n = $n; b = np.random.rand(n, n); a = b.T @ b" "np.linalg.eigh(a)"; done
n=10 | 20000 loops, best of 5: 11.6 usec per loop
n=20 | 5000 loops, best of 5: 47.1 usec per loop
n=50 | 200 loops, best of 5: 1.01 msec per loop
n=100 | 100 loops, best of 5: 2.18 msec per loop
n=200 | 20 loops, best of 5: 13.3 msec per loop
n=500 | 1 loop, best of 5: 23 msec per loop
n=1000 | 1 loop, best of 5: 573 msec per loop
n=2000 | 1 loop, best of 5: 2.38 sec per loop
```
--->

Let's see what happens when we change the number of threads used by NumPy's BLAS backend.

NumPy's BLAS backends use OpenMP for multithreading.
The number of threads used can be controlled by environment variables such as OMP_NUM_THREADS, the generic OpenMP environment variable, or MKL_NUM_THREADS and OPENBLAS_NUM_THREADS, library specific environment variables respectively corresponding to MKL and OpenBLAS.
<!--- or controlled at runtime using a library such as [`threadpoolctl`](https://github.com/joblib/threadpoolctl). --->
<!--- 
The numpy installed in the default conda environment of the python module at NERSC is configured with Intel MKL as the backend.
Intel MKL uses the number of threads equal to the number of physical cores on the system by default.

TODO: consider moving this

```bash
> python -c 'import numpy as np; np.show_config()'
blas_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/global/common/software/nersc/pm-2022q2/sw/python/3.9-anaconda-2021.11/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/global/common/software/nersc/pm-2022q2/sw/python/3.9-anaconda-2021.11/include']
blas_opt_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/global/common/software/nersc/pm-2022q2/sw/python/3.9-anaconda-2021.11/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/global/common/software/nersc/pm-2022q2/sw/python/3.9-anaconda-2021.11/include']
lapack_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/global/common/software/nersc/pm-2022q2/sw/python/3.9-anaconda-2021.11/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/global/common/software/nersc/pm-2022q2/sw/python/3.9-anaconda-2021.11/include']
lapack_opt_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/global/common/software/nersc/pm-2022q2/sw/python/3.9-anaconda-2021.11/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/global/common/software/nersc/pm-2022q2/sw/python/3.9-anaconda-2021.11/include']
Supported SIMD extensions in this NumPy install:
    baseline = SSE,SSE2,SSE3
    found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2
    not found = AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CNL
```
--->
In the following example, we run the same benchmark we ran a moment ago but with different values of OMP_NUM_THREADS.
The `indirect-scaling.sh` bash script runs our benchmark, looping over values of OMP_NUM_THREADS.

```bash
> bash indirect-scaling.sh
nthreads=1 | n=1000 | 2 loops, best of 5: 188 msec per loop
nthreads=2 | n=1000 | 2 loops, best of 5: 126 msec per loop
nthreads=4 | n=1000 | 2 loops, best of 5: 117 msec per loop
nthreads=8 | n=1000 | 2 loops, best of 5: 94.9 msec per loop
nthreads=16 | n=1000 | 2 loops, best of 5: 85.4 msec per loop
nthreads=32 | n=1000 | 5 loops, best of 5: 98.3 msec per loop
nthreads=64 | n=1000 | 1 loop, best of 5: 327 msec per loop
nthreads=128 | n=1000 | 1 loop, best of 5: 468 msec per loop
nthreads=256 | n=1000 | 1 loop, best of 5: 446 msec per loop
```

Our earlier measurement of "427 msec" is right about where we have nthreads=128, the number of physical cores on a Perlmutter CPU node (which is usually the default setting for the number of threads to use).
Note that the performance improves as we increase from nthreads=1 to nthreads=16, starts to degrade at nthreads=32, and then degrades significantly beyond that point.

<!--- TODO
discuss "strong scaling" vs "weak scaling"
--->

<!---
This is getting ahead of ourselves a bit, but we can run the same application on a GPU (using CuPy) which takes about 18.5 msec.
The speedup factor between CPU/GPU would likely improve at larger matrix sizes.

```bash
> python -c 'import cupy as cp; from cupyx.profiler import benchmark; n = 1000; b = cp.random.rand(n, n); a = b.T @ b; print(benchmark(cp.linalg.eigh, (a,), n_repeat=5))'
eigh                :    CPU:16353.855 us   +/-1780.469 (min:13992.920 / max:17809.189) us     GPU-0:18412.544 us   +/-2086.878 (min:15674.368 / max:20120.577) us
```
--->

It's important to keep this indirect parallelism in mind as we begin to directly program with parallelism in Python.

## Embarrassingly Parallel Problems in Python

An embarrassingly parallel problem is one in which there is little or no dependency between subdivisions of the problem.
This is a frequently encountered class of problems encountered by scientists proccessing data from experiments.

Let's consider an example problem where we need to perform the same computation on several input data.
To keep this example simple, we will randomly generate data to process in parallel:

```python
data = list()
for i in range(ntasks):
    np.random.seed(i)
    b = np.random.rand(n, n)
    a = b.T @ b
    data.append(a)
```

We will process individual units of our data with the following function:

```python
def process_data(id, a):
    w, v = np.linalg.eigh(a)
    return id
```

We only return the id of the task but an actual problem might return something useful like a piece of data to be written to a file.
We'll collect the ids of completed tasks and use that to verify we completed all the tasks.
You do not typically have to include this sort of book keeping but it is a simple way for us to verify that our examples are correctly processing all tasks.

We'll also try to ignore indirect parallelism in NumPy by setting `OMP_NUM_THREADS=1` in the examples below.

### Serial `for`-loop 

The serial `for`-loop solution to this problem is implemented in `0-serial.py`.

When we run this example, we measure a wall clock time of about 29 seconds.
The progress bar reports a work rate of 4.48 iterations per second (223 msec per iteration) which roughly corresponds to the performance we measured earlier with nthreads=1.

```bash
> time OMP_NUM_THREADS=1 python 0-serial.py
serial
n=1000 ntasks=128
100%|█████████████████████████████████████████| 128/128 [00:24<00:00,  5.21it/s]
True

real	0m28.969s
user	0m28.464s
sys	0m0.344s
```

5.21 iterations per second (192 ms per iteration) is pretty close to the 190 ms measurement from the previous section.
Note we changed our benchmarking method so this is a reassuring sanity check that we have not introduced a bias in to our benchmark.


### Process-based parallelism with Python's multiprocessing

Now let's write the parallelism directly using the [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) library which is part of the Python Standard Libary.
We'll use a `multiprocessing.Pool` object to spawn a group of worker processes and map our set of tasks onto that group of workers.
By spawning new Python processes, multiprocessing bypasses the GIL limitation that we would encounter using threads within single Python process.

Our next example is `1-multiprocessing.py`. First, let's run this using a single process:

```bash
> time OMP_NUM_THREADS=1 python 1-multiprocessing.py --nproc 1
multiprocessing
n=1000 ntasks=128 nproc=1
100%|█████████████████████████████████████████| 128/128 [00:25<00:00,  5.06it/s]
True

real	0m29.663s
user	0m29.274s
sys	0m0.886s
```

The performance is similar to the previous example, perhaps a little slower due to multiprocessing overhead.
Let's run this again increasing the number of processes:

```
> time OMP_NUM_THREADS=1 python 1-multiprocessing.py --nproc 4
multiprocessing
n=1000 ntasks=128 nproc=4
100%|█████████████████████████████████████████| 128/128 [00:06<00:00, 18.29it/s]
True

real	0m11.752s
user	0m31.486s
sys	0m1.492s
```

Using 4 processes results in nearly a 4x speedup.

Exercise: How does the performance change as you continue increasing the number of processes?

<!--- Answer

It increases but starts to taper off.

Possible explanations:
 * communication or startup overhead of multiple processes
 * resource contention

--->


### MPI-based parallelism with mpi4py

In Python, the most common way to use [MPI](https://docs.nersc.gov/development/programming-models/mpi/) is through the [`mpi4py` library](https://mpi4py.readthedocs.io/en/stable/intro.html).

MPI programs must be launched with a special launcher program.
At NERSC, that is tied into the [`srun` SLURM command](https://docs.nersc.gov/jobs/#srun).
The MPI launcher creates the specified number of processes to run the program.
This is different then the previous example where we launched a single parent process that spawned children processes at some later point.

The MPI version of our example is implemented in `2-mpi4py.py`.

> :warning: warning :warning:
>
> Currently, the mpi4py in the base conda env provided by python module at NERSC requires cudatoolkit, even for CPU-only code.
> For now, you can run simply run `module load cudatoolkit` to satisfy this requirement.
> We'll try to fix this soon.

```bash
> time OMP_NUM_THREADS=1 srun -n 4 -c 2 --cpu-bind=cores python 2-mpi4py.py
mpi4py (nompi=False)
n=1000 ntasks=128 size=4
True

real	0m9.338s
user	0m0.024s
sys	0m0.027s
```

Exercise: Find the optimal number of MPI tasks for this example by changing the value specified by `-n`.

<!--- Answer
Performance peaks around 32 MPI tasks.
--->

<!---
(This is the first time we've used srun. Capture start-up time? mpi init? I assume that is why this is slower than the previous implementations)
--->

### Task-based parallelism with Dask

Here is similar implementation with Dask. See the Dask page in the NERSC docs with more info.

See `3-dask.py`.

```bash
> time OMP_NUM_THREADS=1 python 3-dask.py
dask
n=1000 ntasks=128 nworkers=32 threads_per_worker=1
[########################################] | 100% Completed |  1.4s
True

real	0m4.890s
user	0m54.126s
sys	0m35.516s
```


### GPU-based parallelism with CuPy

Let's hop over to a GPU node and create a new environment with [CuPy](https://docs.cupy.dev/en/stable/overview.html), a NumPy/SciPy-compatible library for GPU-accelerated computing with Python.
The following instructions are based on the instruction in the [Perlmutter Python NERSC documentation](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#guide-to-using-python-on-perlmutter).

```bash 
salloc -C gpu -A <account_g> -N 1 --gpus-per-node 4 -q interactive -t 60

module load PrgEnv-gnu cudatoolkit/11.5 python
conda create -n cupy-demo python=3.9 pip numpy scipy tqdm
conda activate cupy-demo
pip install cupy-cuda115
MPICC="cc -target-accel=nvidia80 -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
```

The CuPy implementation in `4-cupy.py` is nearly identical to serial implementation that we started with.

```bash
> time python 4-cupy.py
cupy
n=1000 ntasks=128 device_pci_bus_id='0000:C3:00.0'
100%|█████████████████████████████████████████| 128/128 [00:02<00:00, 62.25it/s]
True

real	0m3.471s
user	0m6.155s
sys	0m10.462s
```

This is about 10x faster than the CPU version.

Note that `process_data` is identical to the serial-version and uses the NumPy API.
The NumPy API detects that the input data is a CuPy ndarray and dispatches to the call to CuPy's `cupy.linalg.eigh` method.


### Multi-GPU parallelism with CuPy + mpi4py

This example demonstrates how to combine CuPy with mpi4py to share GPU data directly between GPUs without passing through CPU memory.

Note you must set MPICH_GPU_SUPPORT_ENABLED=1 when using CUDA-aware MPI.

```
> time MPICH_GPU_SUPPORT_ENABLED=1 srun -n 4 -c 2 --cpu-bind=cores --gpus-per-node=4 python 5-cupy-mpi4py.py
rank=0 device_pci_bus_id='0000:03:00.0'
rank=3 device_pci_bus_id='0000:C1:00.0'
rank=1 device_pci_bus_id='0000:41:00.0'
rank=2 device_pci_bus_id='0000:81:00.0'
cupy-mpi4py
n=1000 ntasks=128 size=4
Processed 128 tasks in 2.20s (58.30it/s)
True

real	0m4.344s
user	0m0.025s
sys	0m0.012s
```

<!--- TODO

extend multi-gpu examples with multiprocessing/dask to leverage all 4 GPUs

--->

### Multi-node parallism with mp4py

The previous mpi4py example can be reused to scale beyond a single node.

Request an interactive job with two CPU nodes:

```bash
> salloc -C cpu -A nstaff -N 2 -q interactive -t 60
> module load python
```

Launch with two nodes, four MPI tasks per node:

```bash
> time OMP_NUM_THREADS=1 srun -N 2 --ntasks-per-node 4 -c 2 --cpu-bind=cores python 2-mpi4py.py
mpi4py
n=1000 ntasks=128 size=8
Processed 128 tasks in 4.37s (29.26it/s)
True

real	0m11.210s
user	0m0.019s
sys	0m0.023s
```

<!---

## TODO Next steps

Add some links to resources for next steps

 * Profiling
 * numba.cuda
 * numba.pyomp
 * workflow tools
 * CUDA MPS

--->
