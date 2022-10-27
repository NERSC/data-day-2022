import time
from library import estimate_pi

import dask
from dask.distributed import Client, progress

if __name__ == "__main__":

    n = 20_000_000
    p = 4

    client = Client(threads_per_worker=1, n_workers=p)

    futures = []
    for i in range(p):
        futures.append(dask.delayed(estimate_pi)(n//p))

    start = time.time()
    dask.compute(*futures)
    end = time.time()
    print(end - start)

