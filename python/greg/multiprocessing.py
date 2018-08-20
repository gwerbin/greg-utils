import multiprocessing
import signal
from contextlib import contextmanager


# https://stackoverflow.com/a/44869451/2954547
def ignore_sigint():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def multi_process(n, *args, **kwargs):
    with multiprocessing.Pool(n, initializer=ignore_sigint) as pool:
        try:
            yield from pool.imap_unordered(*args, **kwargs)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
