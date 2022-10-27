#!/usr/bin/env python

import argparse

import numpy as np
from tqdm import tqdm

def process_data(id, a):
    w, v = np.linalg.eigh(a)
    return id

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1000)
    parser.add_argument('--ntasks', type=int, default=128)
    args = parser.parse_args()
    n = args.n
    ntasks = args.ntasks

    print('serial')
    print(f'{n=} {ntasks=}')

    # create a random set of data
    data = list()
    for i in range(ntasks):
        np.random.seed(i)
        b = np.random.rand(n, n)
        a = b.T @ b
        data.append(a)

    results = []
    # process tasks
    for i in tqdm(range(ntasks), ncols=80):
        results.append(process_data(i, data[i]))

    # check results
    print(sum(results) == (ntasks * (ntasks - 1)) // 2)
