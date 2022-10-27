#!/bin/bash

n=${1:-1000}
setup="import numpy as np; n = $n; b = np.random.rand(n, n); a = b.T @ b"
for nthreads in 1 2 4 8 16 32 64 128 256; do \
    echo -n "nthreads=$nthreads | n=$n | "
    OMP_NUM_THREADS=$nthreads python -m timeit -s "$setup" "np.linalg.eigh(a)"
done
