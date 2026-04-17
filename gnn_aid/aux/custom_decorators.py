import time
from functools import wraps
import logging
import functools
from typing import Callable

logging.basicConfig(level=logging.INFO)


def retry(
        max_tries: int = 3,
        delay_seconds: int = 1
):
    """
    Allows you to re-execute the program after the Nth amount of time
    :param max_tries: number of restart attempts
    :param delay_seconds: time interval between attempts
    """
    def decorator_retry(
            func: Callable
    ) -> Callable:
        @wraps(func)
        def wrapper_retry(
                *args,
                **kwargs
        ):
            tries = 0
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    if tries == max_tries:
                        raise e
                    time.sleep(delay_seconds)

        return wrapper_retry

    return decorator_retry


def memoize(
        func: Callable
) -> Callable:
    """
    Caching function
    """
    cache = {}

    def wrapper(
            *args
    ):
        if args in cache:
            return cache[args]
        else:
            result = func(*args)
            cache[args] = result
            return result

    return wrapper


def timing_decorator(
        func: Callable
) -> Callable:
    """
    Timing functions
    """
    def wrapper(
            *args,
            **kwargs
    ):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper


def log_execution(
        func: Callable
) -> Callable:
    """
    Function call logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Executing {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Finished executing {func.__name__}")
        return result

    return wrapper
