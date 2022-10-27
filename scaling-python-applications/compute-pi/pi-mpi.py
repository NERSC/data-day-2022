import time
from library import estimate_pi

if __name__ == "__main__":

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    n = 20_000_000
    p = comm.size

    comm.barrier()
    start = time.time()
    estimate_pi(n // p)
    comm.barrier()
    end = time.time()
    
    if comm.rank == 0:
        print(end - start)

