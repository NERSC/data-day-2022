#!/usr/bin/env python

import argparse

import numpy as np

import dask
from dask.distributed import Client, progress


def process_data(id, a):
    w, v = np.linalg.eigh(a)
    return id


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1000)
    parser.add_argument('--ntasks', type=int, default=128)
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--threads-per-worker', type=int, default=1)
    args = parser.parse_args()

    n = args.n
    ntasks = args.ntasks
    nworkers = args.nworkers
    threads_per_worker = args.threads_per_worker

    print(f'dask')
    print(f'{n=} {ntasks=} {nworkers=} {threads_per_worker=}')

    data = list()
    for i in range(ntasks):
        np.random.seed(i)
        b = np.random.rand(n, n)
        a = b.T @ b
        data.append(a)

    client = Client(threads_per_worker=threads_per_worker, n_workers=nworkers)

    # lazy init tasks
    lazy_results = []
    for i in range(ntasks):
        lazy_result = dask.delayed(process_data)(i, data[i])
        lazy_results.append(lazy_result)

    # process tasks
    futures = dask.persist(*lazy_results)
    progress(futures)

    results = dask.compute(*futures)
    print()

    # check results
    print(sum(results) == (ntasks * (ntasks - 1)) // 2)
