import abc
import functools
import concurrent.futures
import numpy as np
import attr


# Abstract base class compatible with both Python 2 and 3.
ABC = abc.ABCMeta('ABC', (object,), {})


def array_copy_immutable(data):
    """Return a copy of data as an immutable numpy array.

    Useful as an attrs converter for frozen classes.
    """
    a = np.copy(data)
    a.flags.writeable = False
    return a


def validate_range(rmin, rmax):
    """attrs validator -- value must be in the interval [rmin, rmax]."""
    def validate(self, attribute, value):
        if not rmin <= value <= rmax:
            raise ValueError(
                f"{attribute.name} must be between {rmin} and {rmax}"
            )
    return validate


def attrib(doc=None, **kwargs):
    """Wrapper for attr.ib with docstring support via the 'doc' argument."""
    if doc is not None:
        metadata = kwargs.setdefault('metadata', {})
        metadata['doc'] = doc
    return attr.ib(**kwargs)


def cached_property(fget):
    """Decorator for a read-only property whose value is only evaluated once."""
    cache_attr = f"_{fget.__name__}"
    def wrapper(self):
        if not hasattr(self, cache_attr):
            object.__setattr__(self, cache_attr, fget(self))
        return getattr(self, cache_attr)
    return property(functools.update_wrapper(wrapper, fget))


def executor_submit(executor, fn, task_args):
    """Submit tasks to a concurrent.futures.Executor, returning Futures.

    All tasks use the same callable `fn`. The *args for each execution of `fn`
    are taken from the iterable `task_args`.

    """
    return [executor.submit(fn, *args) for args in task_args]


def future_results(futures):
    """Return results for a list of futures."""
    return [f.result() for f in futures]


def future_progress(futures, batch_size=1):
    """Return results for a list of futures, with progress reporting."""
    n = len(futures)
    results = []
    for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
        if i % batch_size == 0 or i == n:
            print(f"\r{i}/{n}", end="", flush=True)
        results.append(f.result())
    print()
    return results
