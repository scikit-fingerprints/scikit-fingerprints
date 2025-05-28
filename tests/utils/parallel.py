import re

import pytest
from sklearn.utils.parallel import delayed

from skfp.utils.parallel import ProgressParallel, run_in_parallel


def test_progress_parallel(capsys):
    func = lambda x: x + 1
    data = list(range(100))
    parallel = ProgressParallel(tqdm_settings={"total": len(data)})
    _ = parallel(delayed(func)(num) for num in data)
    stderr = capsys.readouterr().err  # tqdm outputs to stderr

    # example output: 17%|█▋        | 2/12 [00:00<00:00, 458.09it/s]

    # percentage, e.g. 10%
    assert re.search(r"\d+%", stderr)

    # processed iterations, e.g. 1/10
    assert re.search(r"\d+/\d+", stderr)

    # time, e.g. 00:01
    assert re.search(r"\d\d:\d\d", stderr)

    # iterations per second, e.g. 1.23it/s
    assert re.search(r"it/s", stderr)


def test_run_in_parallel():
    func = lambda X: [x + 1 for x in X]
    data = list(range(100))
    result_sequential = func(data)
    result_parallel = run_in_parallel(func, data, n_jobs=-1, flatten_results=True)
    assert result_sequential == result_parallel


def test_run_in_parallel_batch_size():
    func = lambda X: [x + 1 for x in X]
    data = list(range(100))
    result_sequential = func(data)
    result_parallel = run_in_parallel(
        func, data, n_jobs=-1, batch_size=1, flatten_results=True
    )
    assert result_sequential == result_parallel


def test_run_in_parallel_invalid_batch_size():
    func = lambda X: [x + 1 for x in X]
    data = list(range(100))
    with pytest.raises(ValueError) as exc_info:
        run_in_parallel(func, data, n_jobs=-1, batch_size=-1, flatten_results=True)

    assert "batch_size must be positive" in str(exc_info)


def test_run_in_parallel_single_element_func():
    func = lambda x: x + 1
    data = list(range(100))
    result_parallel = run_in_parallel(func, data, n_jobs=-1, single_element_func=True)
    expected_result = list(range(1, 101))
    assert result_parallel == expected_result


def test_run_in_parallel_verbose_dict(capsys):
    func = lambda x: x + 1
    data = list(range(100))
    run_in_parallel(func, data, n_jobs=-1, single_element_func=True, verbose={})
    stderr = capsys.readouterr().err  # tqdm outputs to stderr

    assert "100%" in stderr
    assert "100/100" in stderr
