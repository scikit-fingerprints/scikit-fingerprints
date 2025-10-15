import pytest

from skfp.datasets.moleculeace import (
    load_chembl204_ki,
    load_chembl214_ki,
    load_chembl218_ec50,
    load_chembl219_ki,
    load_chembl228_ki,
    load_chembl231_ki,
    load_chembl233_ki,
    load_chembl234_ki,
    load_chembl235_ec50,
    load_chembl236_ki,
    load_chembl237_ec50,
    load_chembl237_ki,
    load_chembl238_ki,
    load_chembl239_ec50,
    load_chembl244_ki,
    load_chembl262_ki,
    load_chembl264_ki,
    load_chembl287_ki,
    load_chembl1862_ki,
    load_chembl1871_ki,
    load_chembl2034_ki,
    load_chembl2047_ec50,
    load_chembl2147_ki,
    load_chembl2835_ki,
    load_chembl2971_ki,
    load_chembl3979_ec50,
    load_chembl4005_ki,
    load_chembl4203_ki,
    load_chembl4616_ec50,
    load_chembl4792_ki,
    load_moleculeace_dataset,
)
from tests.datasets.test_utils import run_basic_dataset_checks


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize(
    "dataset_name, load_func, expected_length, num_tasks, task_type",
    [
        ("chembl204_ki", load_chembl204_ki, 2754, 1, "regression"),
        ("chembl214_ki", load_chembl214_ki, 3317, 1, "regression"),
        ("chembl218_ec50", load_chembl218_ec50, 1031, 1, "regression"),
        ("chembl219_ki", load_chembl219_ki, 1865, 1, "regression"),
        ("chembl228_ki", load_chembl228_ki, 1704, 1, "regression"),
        ("chembl231_ki", load_chembl231_ki, 973, 1, "regression"),
        ("chembl233_ki", load_chembl233_ki, 3142, 1, "regression"),
        ("chembl234_ki", load_chembl234_ki, 3657, 1, "regression"),
        ("chembl235_ec50", load_chembl235_ec50, 2349, 1, "regression"),
        ("chembl236_ki", load_chembl236_ki, 2598, 1, "regression"),
        ("chembl237_ec50", load_chembl237_ec50, 955, 1, "regression"),
        ("chembl237_ki", load_chembl237_ki, 2603, 1, "regression"),
        ("chembl238_ki", load_chembl238_ki, 1052, 1, "regression"),
        ("chembl239_ec50", load_chembl239_ec50, 1721, 1, "regression"),
        ("chembl244_ki", load_chembl244_ki, 3097, 1, "regression"),
        ("chembl262_ki", load_chembl262_ki, 856, 1, "regression"),
        ("chembl264_ki", load_chembl264_ki, 2862, 1, "regression"),
        ("chembl287_ki", load_chembl287_ki, 1328, 1, "regression"),
        ("chembl1862_ki", load_chembl1862_ki, 794, 1, "regression"),
        ("chembl1871_ki", load_chembl1871_ki, 659, 1, "regression"),
        ("chembl2034_ki", load_chembl2034_ki, 750, 1, "regression"),
        ("chembl2047_ec50", load_chembl2047_ec50, 631, 1, "regression"),
        ("chembl2147_ki", load_chembl2147_ki, 1456, 1, "regression"),
        ("chembl2835_ki", load_chembl2835_ki, 615, 1, "regression"),
        ("chembl2971_ki", load_chembl2971_ki, 976, 1, "regression"),
        ("chembl3979_ec50", load_chembl3979_ec50, 1125, 1, "regression"),
        ("chembl4005_ki", load_chembl4005_ki, 960, 1, "regression"),
        ("chembl4203_ki", load_chembl4203_ki, 731, 1, "regression"),
        ("chembl4616_ec50", load_chembl4616_ec50, 682, 1, "regression"),
        ("chembl4792_ki", load_chembl4792_ki, 1471, 1, "regression"),
    ],
)
def test_load_dataset(dataset_name, load_func, expected_length, num_tasks, task_type):
    smiles_list, y = load_func()
    df = load_moleculeace_dataset(dataset_name, as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=expected_length,
        num_tasks=num_tasks,
        task_type=task_type,
    )
