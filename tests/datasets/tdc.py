import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skfp.datasets.tdc import load_tdc_benchmark, load_tdc_splits
from skfp.datasets.tdc.adme.approved_pampa_ncats import load_approved_pampa_ncats
from skfp.datasets.tdc.adme.b3db_classification import load_b3db_classification
from skfp.datasets.tdc.adme.b3db_regression import load_b3db_regression
from skfp.datasets.tdc.adme.bbb_martins import load_bbb_martins
from skfp.datasets.tdc.adme.bioavailability_ma import load_bioavailability_ma
from skfp.datasets.tdc.adme.caco2_wang import load_caco2_wang
from skfp.datasets.tdc.adme.clearance_hepatocyte_az import load_clearance_hepatocyte_az
from skfp.datasets.tdc.adme.clearance_microsome_az import load_clearance_microsome_az
from skfp.datasets.tdc.adme.cyp1a2_veith import load_cyp1a2_veith
from skfp.datasets.tdc.adme.cyp2c9_substrate_carbonmangels import (
    load_cyp2c9_substrate_carbonmangels,
)
from skfp.datasets.tdc.adme.cyp2c9_veith import load_cyp2c9_veith
from skfp.datasets.tdc.adme.cyp2c19_veith import load_cyp2c19_veith
from skfp.datasets.tdc.adme.cyp2d6_substrate_carbonmangels import (
    load_cyp2d6_substrate_carbonmangels,
)
from skfp.datasets.tdc.adme.cyp2d6_veith import load_cyp2d6_veith
from skfp.datasets.tdc.adme.cyp3a4_substrate_carbonmangels import (
    load_cyp3a4_substrate_carbonmangels,
)
from skfp.datasets.tdc.adme.cyp3a4_veith import load_cyp3a4_veith
from skfp.datasets.tdc.adme.half_life_obach import load_half_life_obach
from skfp.datasets.tdc.adme.hia_hou import load_hia_hou
from skfp.datasets.tdc.adme.hlm import load_hlm
from skfp.datasets.tdc.adme.pampa_ncats import load_pampa_ncats
from skfp.datasets.tdc.adme.pgp_broccatelli import load_pgp_broccatelli
from skfp.datasets.tdc.adme.ppbr_az import load_ppbr_az
from skfp.datasets.tdc.adme.rlm import load_rlm
from skfp.datasets.tdc.adme.solubility_aqsoldb import load_solubility_aqsoldb
from skfp.datasets.tdc.adme.vdss_lombardo import load_vdss_lombardo
from skfp.datasets.tdc.hts.sarscov2_3clpro_diamond import load_sarscov2_3clpro_diamond
from skfp.datasets.tdc.hts.sarscov2_vitro_touret import load_sarscov2_vitro_touret
from skfp.datasets.tdc.tox.ames import load_ames
from skfp.datasets.tdc.tox.carcinogens_lagunin import load_carcinogens_lagunin
from skfp.datasets.tdc.tox.dili import load_dili
from skfp.datasets.tdc.tox.herg import load_herg
from skfp.datasets.tdc.tox.herg_central_at_1um import load_herg_central_at_1um
from skfp.datasets.tdc.tox.herg_central_at_10um import load_herg_central_at_10um
from skfp.datasets.tdc.tox.herg_central_inhib import load_herg_central_inhib
from skfp.datasets.tdc.tox.herg_karim import load_herg_karim
from skfp.datasets.tdc.tox.ld50_zhu import load_ld50_zhu
from skfp.datasets.tdc.tox.skin_reaction import load_skin_reaction
from tests.datasets.test_utils import run_basic_dataset_checks


def get_dataset_names() -> list[str]:
    return [
        # adme
        "approved_pampa_ncats",
        "b3db_classification",
        "b3db_regression",
        "bbb_martins",
        "bioavailability_ma",
        "caco2_wang",
        "clearance_hepatocyte_az",
        "clearance_microsome_az",
        "cyp1a2_veith",
        "cyp2c19_veith",
        "cyp2c9_substrate_carbonmangels",
        "cyp2c9_veith",
        "cyp2d6_substrate_carbonmangels",
        "cyp2d6_veith",
        "cyp3a4_substrate_carbonmangels",
        "cyp3a4_veith",
        "half_life_obach",
        "hia_hou",
        "hlm",
        "pampa_ncats",
        "pgp_broccatelli",
        "ppbr_az",
        "rlm",
        "solubility_aqsoldb",
        "vdss_lombardo",
        # hts
        "sarscov2_3clpro_diamond",
        "sarscov2_vitro_touret",
        # tdc
        "ames",
        "carcinogens_lagunin",
        "dili",
        "herg",
        "herg_central_at_10um",
        "herg_central_at_1um",
        "herg_central_inhib",
        "herg_karim",
        "ld50_zhu",
        "skin_reaction",
    ]


def test_load_tdc_benchmark():
    dataset_names = get_dataset_names()
    benchmark_full = load_tdc_benchmark(as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert benchmark_names == dataset_names


def test_load_tdc_benchmark_subset():
    dataset_names = ["approved_pampa_ncats", "sarscov2_3clpro_diamond", "ames"]
    benchmark_full = load_tdc_benchmark(subset=dataset_names, as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert benchmark_names == dataset_names


def test_load_tdc_benchmark_wrong_subset():
    dataset_names = ["approved_pampa_ncats", "Nonexistent"]
    with pytest.raises(ValueError) as exc_info:
        load_tdc_benchmark(subset=dataset_names, as_frames=True)

    assert "Dataset name 'Nonexistent' not recognized" in str(exc_info)


@pytest.mark.parametrize("dataset_name", get_dataset_names())
def test_load_ogb_splits(dataset_name):
    train, valid, test = load_tdc_splits(dataset_name)
    assert isinstance(train, list)
    assert len(train) > 0
    assert all(isinstance(idx, int) for idx in train)

    assert isinstance(valid, list)
    assert len(valid) > 0
    assert all(isinstance(idx, int) for idx in valid)

    assert isinstance(test, list)
    assert len(test) > 0
    assert all(isinstance(idx, int) for idx in test)

    assert len(train) > len(valid)
    assert len(train) > len(test)


@pytest.mark.parametrize("dataset_name", get_dataset_names())
def test_load_ogb_splits_as_dict(dataset_name):
    train, valid, test = load_tdc_splits(dataset_name)
    split_idxs = load_tdc_splits(dataset_name, as_dict=True)
    assert isinstance(split_idxs, dict)
    assert set(split_idxs.keys()) == {"train", "valid", "test"}
    assert split_idxs["train"] == train
    assert split_idxs["valid"] == valid
    assert split_idxs["test"] == test


@pytest.mark.parametrize(
    "dataset_name, dataset_length",
    [
        ("herg", 655),
        ("herg_central_at_1um", 306893),
        ("herg_central_at_10um", 306893),
        ("herg_central_inhib", 306893),
        ("herg_karim", 13445),
        ("dili", 475),
        ("skin_reaction", 404),
        ("ames", 7278),
        ("carcinogens_lagunin", 280),
        ("ld50_zhu", 7385),
        ("solubility_aqsoldb", 9982),
        ("caco2_wang", 910),
        ("pampa_ncats", 2034),
        ("approved_pampa_ncats", 142),
        ("hia_hou", 578),
        ("pgp_broccatelli", 1218),
        ("bioavailability_ma", 640),
        ("vdss_lombardo", 1130),
        ("cyp2c19_veith", 12665),
        ("cyp2d6_veith", 13130),
        ("cyp3a4_veith", 12328),
        ("cyp1a2_veith", 12579),
        ("cyp2c9_veith", 12092),
        ("cyp2c9_substrate_carbonmangels", 669),
        ("cyp2d6_substrate_carbonmangels", 667),
        ("cyp3a4_substrate_carbonmangels", 670),
        ("bbb_martins", 2030),
        ("b3db_classification", 6167),
        ("b3db_regression", 942),
        ("ppbr_az", 1614),
        ("half_life_obach", 667),
        ("clearance_hepatocyte_az", 1213),
        ("clearance_microsome_az", 1102),
        ("hlm", 6013),
        ("rlm", 5590),
        ("sarscov2_3clpro_diamond", 880),
        ("sarscov2_vitro_touret", 1484),
    ],
)
def test_load_tdc_splits_lengths(dataset_name, dataset_length):
    train, valid, test = load_tdc_splits(dataset_name)
    loaded_length = len(train) + len(valid) + len(test)
    assert loaded_length == dataset_length


def test_load_tdc_splits_nonexistent_dataset():
    with pytest.raises(InvalidParameterError) as error:
        load_tdc_splits("nonexistent")

    print(str(error.value))

    assert str(error.value).startswith(
        "The 'dataset_name' parameter of load_tdc_splits must be a str among"
    )


@pytest.mark.parametrize(
    "dataset_name, load_func, expected_length, num_tasks, task_type",
    [
        ("herg", load_herg, 655, 1, "binary_classification"),
        ("herg_central_at_1um", load_herg_central_at_1um, 306893, 1, "regression"),
        ("herg_central_at_10um", load_herg_central_at_10um, 306893, 1, "regression"),
        (
            "herg_central_inhib",
            load_herg_central_inhib,
            306893,
            1,
            "binary_classification",
        ),
        ("herg_karim", load_herg_karim, 13445, 1, "binary_classification"),
        ("dili", load_dili, 475, 1, "binary_classification"),
        ("skin_reaction", load_skin_reaction, 404, 1, "binary_classification"),
        ("ames", load_ames, 7278, 1, "binary_classification"),
        (
            "carcinogens_lagunin",
            load_carcinogens_lagunin,
            280,
            1,
            "binary_classification",
        ),
        ("ld50_zhu", load_ld50_zhu, 7385, 1, "regression"),
        ("solubility_aqsoldb", load_solubility_aqsoldb, 9982, 1, "regression"),
        ("caco2_wang", load_caco2_wang, 910, 1, "regression"),
        ("pampa_ncats", load_pampa_ncats, 2034, 1, "binary_classification"),
        (
            "approved_pampa_ncats",
            load_approved_pampa_ncats,
            142,
            1,
            "binary_classification",
        ),
        ("hia_hou", load_hia_hou, 578, 1, "binary_classification"),
        ("pgp_broccatelli", load_pgp_broccatelli, 1218, 1, "binary_classification"),
        (
            "bioavailability_ma",
            load_bioavailability_ma,
            640,
            1,
            "binary_classification",
        ),
        ("vdss_lombardo", load_vdss_lombardo, 1130, 1, "regression"),
        ("cyp2c19_veith", load_cyp2c19_veith, 12665, 1, "binary_classification"),
        ("cyp2d6_veith", load_cyp2d6_veith, 13130, 1, "binary_classification"),
        ("cyp3a4_veith", load_cyp3a4_veith, 12328, 1, "binary_classification"),
        ("cyp1a2_veith", load_cyp1a2_veith, 12579, 1, "binary_classification"),
        ("cyp2c9_veith", load_cyp2c9_veith, 12092, 1, "binary_classification"),
        (
            "cyp2c9_substrate_carbonmangels",
            load_cyp2c9_substrate_carbonmangels,
            669,
            1,
            "binary_classification",
        ),
        (
            "cyp2d6_substrate_carbonmangels",
            load_cyp2d6_substrate_carbonmangels,
            667,
            1,
            "binary_classification",
        ),
        (
            "cyp3a4_substrate_carbonmangels",
            load_cyp3a4_substrate_carbonmangels,
            670,
            1,
            "binary_classification",
        ),
        ("bbb_martins", load_bbb_martins, 2030, 1, "binary_classification"),
        (
            "b3db_classification",
            load_b3db_classification,
            6167,
            1,
            "binary_classification",
        ),
        ("b3db_regression", load_b3db_regression, 942, 1, "regression"),
        ("ppbr_az", load_ppbr_az, 1614, 1, "regression"),
        ("half_life_obach", load_half_life_obach, 667, 1, "regression"),
        (
            "clearance_hepatocyte_az",
            load_clearance_hepatocyte_az,
            1213,
            1,
            "regression",
        ),
        ("clearance_microsome_az", load_clearance_microsome_az, 1102, 1, "regression"),
        ("hlm", load_hlm, 6013, 1, "binary_classification"),
        ("rlm", load_rlm, 5590, 1, "binary_classification"),
        (
            "sarscov2_3clpro_diamond",
            load_sarscov2_3clpro_diamond,
            880,
            1,
            "binary_classification",
        ),
        (
            "sarscov2_vitro_touret",
            load_sarscov2_vitro_touret,
            1484,
            1,
            "binary_classification",
        ),
    ],
)
def test_load_dataset(dataset_name, load_func, expected_length, num_tasks, task_type):
    smiles_list, y = load_func()
    df = load_func(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=expected_length,
        num_tasks=num_tasks,
        task_type=task_type,
    )
