import dask.bag as db


def f(x):
    return x**2


def pred(x):
    return x % 2 == 0


def test_dask_bags():
    bag = db.from_sequence(range(6))
    x = bag.map(f)
    assert x.compute() == [0, 1, 4, 9, 16, 25]
    x = bag.filter(pred)
    assert x.compute() == [0, 2, 4]
