from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

from joblib import effective_n_jobs
from sklearn.utils.parallel import Parallel, delayed
from tqdm import tqdm


class ProgressParallel(Parallel):
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
    data: Sequence[Any],
    batch_size: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
) -> list[Any]:
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

    return results
