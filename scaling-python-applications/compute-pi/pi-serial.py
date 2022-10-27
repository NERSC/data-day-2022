import time
from library import estimate_pi

n = 20_000_000

start = time.time()
result = estimate_pi(n)
end = time.time()

print(end - start)
