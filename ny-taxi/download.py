"""Script to downlad NY Taxi data.

- Uses `pooch` to manage and download files from DATA_BASE_URL
- TRIP_DATA_NAME is the name of the subdirectory where the parquet
files are stored.
"""

import pooch
from pathlib import Path


DATA_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
TRIP_DATA_NAME = "trip-data"
REGISTRY_FILE = f"{TRIP_DATA_NAME}-registry.txt"

def taxi_filename(year: int = 2025, month: int = 1):
    return f"yellow_tripdata_{year:04}-{month:02}.parquet"


def bootstrap_taxi_data(
    path: Path, year: int = 2025
) -> pooch.Pooch:
    return pooch.create(
        path=path,
        base_url=DATA_BASE_URL,
        registry={
            taxi_filename(year=year, month=month): None for month in range(1, 13)
        },
    )


def download_all(p: pooch.Pooch):
    for filename in p.registry.keys():
        p.fetch(filename)


def make_registry(p: pooch.Pooch):
    path = Path(p.path)
    pooch.make_registry(path, path.parent / REGISTRY_FILE)


def taxi_data(
    data_path: Path,
) -> pooch.Pooch:
    registry = data_path /  REGISTRY_FILE

    p = pooch.create(
        path=data_path / TRIP_DATA_NAME,
        base_url=DATA_BASE_URL,
        registry=None,
    )
    p.load_registry(registry)
    return p


def main():
    nyt_path = Path(__file__).parent
    data_dir = "data"
    registry = nyt_path / data_dir / REGISTRY_FILE

    if registry.exists():
        p = taxi_data(data_path = nyt_path / data_dir)
        download_all(p)
    else:
        p = bootstrap_taxi_data(nyt_path / data_dir / TRIP_DATA_NAME)
        download_all(p)
        make_registry(p)


if __name__ == "__main__":
    main()
# TODO: add unit test that data can be read
