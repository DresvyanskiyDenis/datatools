import functools
import time

# time calculation of the function
def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Function {func.__name__!r} has been executed in {run_time:.4f} secs")
        return value
    return wrapper_timer