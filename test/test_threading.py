import queue
import threading
import numba
import random


def test_threading():
    input_range = [10**6]*4
    ncpus = 4

    @numba.jit(nopython=True, nogil=True)
    def calc_pi_numba(N):
        M = 0
        for i in range(N):
            # Simulate impact coordinates
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)

            # True if impact happens inside the circle
            if x**2 + y**2 < 1.0:
                M += 1
        return 4 * M / N

    # We need to define a worker function that fetches jobs from the queue.
    def worker(q):
        while True:
            try:
                x = q.get(block=False)
                print(calc_pi_numba(x), end=' ', flush=True)
            except queue.Empty:
                break

    # Create the queue, and fill it with input values
    work_queue = queue.Queue()
    for i in input_range:
        work_queue.put(i)

    # Start a number of threads
    threads = [
        threading.Thread(target=worker, args=(work_queue,))
        for i in range(ncpus)]

    for t in threads:
        t.start()

    # Wait until all of them are done
    for t in threads:
        t.join()

    print()
