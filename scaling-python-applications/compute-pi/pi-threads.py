from threading import Thread
import time
import sys
from library import estimate_pi

n = 20_000_000
p = 4

t = [
    Thread(target=estimate_pi, args=(n//p,)) 
    for i in range(p)
]

start = time.time()
[t[i].start() for i in range(p)]
[t[i].join() for i in range(p)]
end = time.time()

print(end - start)


