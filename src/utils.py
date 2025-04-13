from time import perf_counter


def time_model(func):
    """Decorator to measure execution time of a function."""

    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        return result, elapsed

    return wrapper
