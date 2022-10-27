import multiprocessing as mp
import time
from library import estimate_pi

if __name__ == "__main__":
    n = 20_000_000
    p = 4

    mp.set_start_method("spawn")
    start = time.time()
    with mp.Pool(processes=p) as pool:
        results = pool.map(estimate_pi, [n//p] * p)
    end = time.time()
    print(end - start)

