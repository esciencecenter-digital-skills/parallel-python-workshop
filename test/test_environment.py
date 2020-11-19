import numpy as np
import scipy
import dask
import snakemake
import numba
from packaging.version import parse as version


def test_library_versions():
    assert version(np.__version__) >= version("1.15")
    assert version(scipy.__version__) >= version("1.5")
    assert version(dask.__version__) >= version("2.20")
    assert version(snakemake.__version__) >= version("5.28")
    assert version(numba.__version__) >= version("0.51")