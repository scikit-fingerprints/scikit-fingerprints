import itertools
from collections.abc import Sequence
from typing import Callable, Optional

from joblib import effective_n_jobs
from sklearn.utils.parallel import Parallel, delayed
from tqdm import tqdm


class ProgressParallel(Parallel):
    """
    A more verbose version of ``joblib.Parallel``, which outputs progress bar.

    Parameters
    ----------
    total : int, default=None
        Total number of inputs to process in parallel.
    """

    def __init__(self, total: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total = total

    def __call__(self, *args, **kwargs):
        with tqdm(total=self.total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self) -> None:
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def run_in_parallel(
    func: Callable,
    data: Sequence,
    n_jobs: Optional[int] = None,
    batch_size: Optional[int] = None,
    flatten_results: bool = False,
    verbose: int = 0,
) -> list:
    """Run a function in parallel on provided data in batches, using joblib.

    Results are returned in the same order as input data. ``func`` function must take
    batch of data, e.g. list of integers, not a single integer.

    If ``func`` returns lists, the result will be a list of lists. To get a flat list
    of results, use ``flatten_results=True`.

    Note that progress bar for ``verbose`` option tracks processing of data batches,
    not individual data points.

    Parameters
    ----------
    func : Callable
        The function to run in parallel. It must take only a single argument,
        a batch of data.

    data : {sequence, array-like} of shape (n_samples,)
        Sequence containing data to process.

    n_jobs : int, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    flatten_results : bool, default=False
        Whether to flatten the results, e.g. to change list of lists of integers
        into a list of integers.

    verbose : int, default=0
        Controls the verbosity. If higher than zero, progress bar will be shown,
        tracking the processing of batches.

    Returns
    -------
    X : list of length (n_samples,)
        The processed data. If processing function returns functions, this will be
        a list of lists.

    Examples
    --------
    >>> from skfp.parallel import run_in_parallel
    >>> func = lambda X: [x + 1 for x in X]
    >>> data = list(range(10))
    >>> run_in_parallel(func, data, n_jobs=-1, batch_size=1)
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    >>> run_in_parallel(func, data, n_jobs=-1, batch_size=1, flatten_results=True)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    n_jobs = effective_n_jobs(n_jobs)

    if batch_size is None:
        batch_size = max(len(data) // n_jobs, 1)
    elif batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    data_batch_gen = (data[i : i + batch_size] for i in range(0, len(data), batch_size))
    num_batches = len(data) // batch_size

    if verbose > 0:
        parallel = ProgressParallel(n_jobs=n_jobs, total=num_batches)
    else:
        parallel = Parallel(n_jobs=n_jobs)

    results = parallel(delayed(func)(data_batch) for data_batch in data_batch_gen)

    if flatten_results:
        results = list(itertools.chain.from_iterable(results))

    return results
