
import os
import pytest
import polars as pl
from pathlib import Path


@pytest.fixture
def data_info():
    data_dir = Path("ny-taxi/data/trip-data/")
    parquet_files = list(data_dir.glob("*.parquet"))
    return data_dir, parquet_files

@pytest.mark.skipif(os.getenv("CI") is not None, reason="data not downloaded on CI")
def test_files_exist(data_info):
    data_dir, parquet_files = data_info
    assert data_dir.exists(), "Missing data directory."

    n_files_expected = 12
    assert len(parquet_files) == n_files_expected, "Not the right number of parquet files"

@pytest.mark.skipif(os.getenv("CI") is not None, reason="data not downloaded on CI")
def test_single_file(data_info):

    _ , parquet_files = data_info

    first_df = pl.read_parquet(parquet_files[0], n_rows=1_000)
    some_column_names = ["fare_amount", "trip_distance", "congestion_surcharge"]
    assert set(some_column_names).issubset(set(first_df.columns)), "Expected column names not present"

