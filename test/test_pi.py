import numpy as np
import numba
import random


def calc_pi(N):
    M = 0
    for i in range(N):
        # Simulate impact coordinates
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        # True if impact happens inside the circle
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N


def calc_pi_numpy(N):
    # Simulate impact coordinates
    pts = np.random.uniform(-1, 1, (2, N))
    # Count number of impacts inside the circle
    M = np.count_nonzero((pts**2).sum(axis=0) < 1)
    return 4 * M / N


@numba.jit
def sum_range_numba(a: int):
    """Compute the sum of the numbers in the range [0, a)."""
    x = 0
    for i in range(a):
        x += i
    return x


@numba.jit
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


def test_calc_pi():
    assert round(calc_pi(10**6)) == 3


def test_calc_pi_numpy():
    assert round(calc_pi_numpy(10**6)) == 3


def test_sum_range_numba():
    for n in np.random.randint(1000, 10000, size=10):
        assert sum_range_numba(n) == (n * (n - 1)) // 2


def test_calc_pi_numba():
    assert round(calc_pi_numba(10**6)) == 3
