"""TODO"""
import functools
import time

from .log import LOGGER

def logging(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        LOGGER.info("Calling %s(%s)", func.__name__, signature)
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        LOGGER.info("%s returned %s", func.__name__, value)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        LOGGER.info("Finished %s in %.4f secs", func.__name__, run_time)
        return value
    return wrapper_debug
