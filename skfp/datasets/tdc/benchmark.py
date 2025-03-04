import os
from collections.abc import Iterator
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.datasets.utils import fetch_splits

from .adme.approved_pampa_ncats import load_approved_pampa_ncats
from .adme.b3db_classification import load_b3db_classification
from .adme.b3db_regression import load_b3db_regression
from .adme.bbb_martins import load_bbb_martins
from .adme.bioavailability_ma import load_bioavailability_ma
from .adme.caco2_wang import load_caco2_wang
from .adme.clearance_hepatocyte_az import load_clearance_hepatocyte_az
from .adme.clearance_microsome_az import load_clearance_microsome_az
from .adme.cyp1a2_veith import load_cyp1a2_veith
from .adme.cyp2c9_substrate_carbonmangels import load_cyp2c9_substrate_carbonmangels
from .adme.cyp2c9_veith import load_cyp2c9_veith
from .adme.cyp2c19_veith import load_cyp2c19_veith
from .adme.cyp2d6_substrate_carbonmangels import load_cyp2d6_substrate_carbonmangels
from .adme.cyp2d6_veith import load_cyp2d6_veith
from .adme.cyp3a4_substrate_carbonmangels import load_cyp3a4_substrate_carbonmangels
from .adme.cyp3a4_veith import load_cyp3a4_veith
from .adme.half_life_obach import load_half_life_obach
from .adme.hia_hou import load_hia_hou
from .adme.hlm import load_hlm
from .adme.pampa_ncats import load_pampa_ncats
from .adme.pgp_broccatelli import load_pgp_broccatelli
from .adme.ppbr_az import load_ppbr_az
from .adme.rlm import load_rlm
from .adme.solubility_aqsoldb import load_solubility_aqsoldb
from .adme.vdss_lombardo import load_vdss_lombardo
from .hts.sarscov2_3clpro_diamond import load_sarscov2_3clpro_diamond
from .hts.sarscov2_vitro_touret import load_sarscov2_vitro_touret
from .tox import (
    load_ames,
    load_carcinogens_lagunin,
    load_dili,
    load_herg,
    load_herg_central_at_1um,
    load_herg_central_at_10um,
    load_herg_central_inhib,
    load_herg_karim,
    load_ld50_zhu,
    load_skin_reaction,
)


@validate_params(
    {
        "subset": [
            None,
            StrOptions({"adme", "hts", "tox"}),
            list,
        ],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_tdc_benchmark(
    subset: Optional[Union[str, list[str]]] = None,
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frames: bool = False,
    verbose: bool = False,
) -> Union[
    Iterator[tuple[str, pd.DataFrame]], Iterator[tuple[str, list[str], np.ndarray]]
]:
    """

    Parameters
    ----------
    subset : {None, "adme", "hts", "tox"} or list of strings
        If ``None``, returns all datasets. String loads only a given subset of all
        datasets. List of strings loads only datasets with given names.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frames : bool, default=False
        If True, returns the raw DataFrame for each dataset. Otherwise, returns SMILES
        as a list of strings, and labels as a NumPy array for each dataset.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : generator of pd.DataFrame or tuples (list[str], np.ndarray)
        Loads and returns datasets with a generator. Returned types depend on the
        ``as_frame`` parameter, either:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    """
    dataset_names = _subset_to_dataset_names(subset)

    dataset_name_to_func = {
        "approved_pampa_ncats": load_approved_pampa_ncats,
        "b3db_classification": load_b3db_classification,
        "b3db_regression": load_b3db_regression,
        "bbb_martins": load_bbb_martins,
        "bioavailability_ma": load_bioavailability_ma,
        "caco2_wang": load_caco2_wang,
        "clearance_hepatocyte_az": load_clearance_hepatocyte_az,
        "clearance_microsome_az": load_clearance_microsome_az,
        "cyp1a2_veith": load_cyp1a2_veith,
        "cyp2c19_veith": load_cyp2c19_veith,
        "cyp2c9_substrate_carbonmangels": load_cyp2c9_substrate_carbonmangels,
        "cyp2c9_veith": load_cyp2c9_veith,
        "cyp2d6_substrate_carbonmangels": load_cyp2d6_substrate_carbonmangels,
        "cyp2d6_veith": load_cyp2d6_veith,
        "cyp3a4_substrate_carbonmangels": load_cyp3a4_substrate_carbonmangels,
        "cyp3a4_veith": load_cyp3a4_veith,
        "half_life_obach": load_half_life_obach,
        "hia_hou": load_hia_hou,
        "hlm": load_hlm,
        "pampa_ncats": load_pampa_ncats,
        "pgp_broccatelli": load_pgp_broccatelli,
        "ppbr_az": load_ppbr_az,
        "rlm": load_rlm,
        "solubility_aqsoldb": load_solubility_aqsoldb,
        "vdss_lombardo": load_vdss_lombardo,
        # hts
        "sarscov2_3clpro_diamond": load_sarscov2_3clpro_diamond,
        "sarscov2_vitro_touret": load_sarscov2_vitro_touret,
        # tox
        "ames": load_ames,
        "carcinogens_lagunin": load_carcinogens_lagunin,
        "dili": load_dili,
        "herg": load_herg,
        "herg_central_at_10um": load_herg_central_at_10um,
        "herg_central_at_1um": load_herg_central_at_1um,
        "herg_central_inhib": load_herg_central_inhib,
        "herg_karim": load_herg_karim,
        "ld50_zhu": load_ld50_zhu,
        "skin_reaction": load_skin_reaction,
    }

    dataset_functions = [dataset_name_to_func[name] for name in dataset_names]

    if as_frames:
        # generator of tuples (dataset_name, DataFrame)
        datasets = (
            (dataset_name, load_function(data_dir, as_frame=True, verbose=verbose))
            for dataset_name, load_function in zip(dataset_names, dataset_functions)
        )
    else:
        # generator of tuples (dataset_name, SMILES, y)
        datasets = (
            (dataset_name, *load_function(data_dir, as_frame=False, verbose=verbose))
            for dataset_name, load_function in zip(dataset_names, dataset_functions)
        )

    return datasets


@validate_params(
    {
        "dataset_name": [
            StrOptions(
                {
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
                    # tox
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
                }
            )
        ],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_tdc_splits(
    dataset_name: str,
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_dict: bool = False,
    verbose: bool = False,
) -> Union[tuple[list[int], list[int], list[int]], dict[str, list[int]]]:
    """
    Parameters
    ----------
    dataset_name
        Name of the dataset to loads splits for.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_dict : bool, default=False
        If True, returns the splits as dictionary with keys "train", "valid" and "test",
        and index lists as values. Otherwise, returns three lists with splits indexes.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : tuple(list[int], list[int], list[int]) or dict
        Depending on the `as_dict` argument, one of:
        - three lists of integer indexes
        - dictionary with "train", "valid" and "test" keys, and values as lists with
        splits indexes

    """
    splits = fetch_splits(
        data_dir,
        dataset_name=f"TDC_{dataset_name}",
        filename=f"tdc_{dataset_name.lower()}_splits.json",
        verbose=verbose,
    )
    if as_dict:
        return splits
    else:
        return splits["train"], splits["valid"], splits["test"]


def _subset_to_dataset_names(subset: Union[str, list[str], None]) -> list[str]:
    # transform given subset (e.g. "adme", "hts" or "tox") into list of dataset names
    # for appropriate TDC datasets

    adme_names = [
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
    ]

    hts_names = [
        "sarscov2_3clpro_diamond",
        "sarscov2_vitro_touret",
    ]

    tox_names = [
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

    all_dataset_names = adme_names + hts_names + tox_names

    if subset is None:
        dataset_names = all_dataset_names
    elif subset == "adme":
        dataset_names = adme_names
    elif subset == "hts":
        dataset_names = hts_names
    elif subset == "tox":
        dataset_names = tox_names
    elif isinstance(subset, (list, set, tuple)):
        for name in subset:
            if name not in all_dataset_names:
                raise ValueError(
                    f"Dataset name '{name}' not recognized among TDC datasets"
                    f"Some TDC datasets can be imported from MoleculeNet benchmark"
                )
        dataset_names = subset
    else:
        raise ValueError(
            f'Value "{subset}" for subset not recognized, must be one of: '
            f'"adme", "hts", or "tox", alternatively '
            f"be a list of strings with dataset names from TDC to load"
        )

    return dataset_names
