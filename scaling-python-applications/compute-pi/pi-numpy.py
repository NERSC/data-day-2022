import time
import numpy as np

def estimate_pi(n):
    x = np.random.random((n, 2))
    return 4.0 * np.mean(np.linalg.norm(x, axis=1) < 1)

n = 20_000_000

start = time.time()
result = estimate_pi(n)
end = time.time()

print(end - start)

