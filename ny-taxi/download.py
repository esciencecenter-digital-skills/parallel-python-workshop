import pooch
from pathlib import Path


def taxi_filename(year: int = 2025, month: int = 1):
    return f"yellow_tripdata_{year:04}-{month:02}.parquet"


def bootstrap_taxi_data(
    path: Path = Path() / "data" / "trip-data", year: int = 2025
) -> pooch.Pooch:
    return pooch.create(
        path=path,
        base_url="https://d37ci6vzurychx.cloudfront.net/trip-data/",
        registry={
            taxi_filename(year=year, month=month): None for month in range(1, 13)
        },
    )


def download_all(p: pooch.Pooch):
    for filename in p.registry.keys():
        p.fetch(filename)


def make_registry(p: pooch.Pooch):
    path = Path(p.path)
    registry_name = path.name + "-registry.txt"
    pooch.make_registry(path, path.parent / registry_name)


def taxi_data(
    data_path: Path = Path() / "data", name: str = "trip-data"
) -> pooch.Pooch:
    path = data_path / name
    registry = data_path / (name + "-registry.txt")

    p = pooch.create(
        path=path,
        base_url="https://d37ci6vzurychx.cloudfront.net/trip-data/",
        registry=None,
    )
    p.load_registry(registry)
    return p


def main():
    registry = Path() / "data" / "trip-data-registry.txt"

    if registry.exists():
        p = taxi_data()
        download_all(p)
    else:
        p = bootstrap_taxi_data()
        download_all(p)
        make_registry(p)


if __name__ == "__main__":
    main()

