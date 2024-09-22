from collections.abc import Sequence
from numbers import Integral
from typing import Any, Optional, Union

import pandas as pd
from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.SimDivFilters.rdSimDivPickers import LeaderPicker
from scipy.spatial.distance import jaccard
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params

from skfp.fingerprints import ECFPFingerprint
from skfp.model_selection.splitters.utils import (
    ensure_nonempty_subset,
    get_data_from_indices,
    split_additional_data,
    validate_train_test_split_sizes,
    validate_train_valid_test_split_sizes,
)
from skfp.utils.validators import ensure_mols


@validate_params(
    {
        "data": ["array-like"],
        "additional_data": ["tuple"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "threshold": [Interval(RealNotInt, 0, 1, closed="both")],
        "return_indices": ["boolean"],
        "n_jobs": [Integral, None],
    }
)
def butina_train_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    test_size: Optional[float] = None,
    threshold: float = 0.65,
    return_indices: bool = False,
    n_jobs: Optional[int] = None,
) -> Union[
    tuple[
        Sequence[Union[str, Mol]], Sequence[Union[str, Mol]], Sequence[Sequence[Any]]
    ],
    tuple[Sequence, ...],
    tuple[Sequence[int], Sequence[int]],
]:
    """
    Split using Taylor-Butina clustering.

    This split uses deterministically partitioned clusters of molecules from Taylor-Butina
    clustering [1]_. It aims to verify the model generalization to structurally novel
    molecules.

    First, molecules are vectorized using binary ECFP4 fingerprint (radius 2) with
    2048 bits. They are then clustered using Leader Clustering, a variant of Taylor-Butina
    clustering by Roger Sayle [2]_ for RDKit. Cluster centroids (central molecules) are
    guaranteed to have at least a given Tanimoto distance between them, as defined by
    `threshold` parameter.

    Clusters are divided deterministically, with the smallest clusters assigned to the
    test subset and the rest to the training subset.

    The split fractions (train_size, test_size) must sum to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit `Mol` objects.

    additional_data: list[sequence]
        Additional sequences to be split alongside the main data (e.g., labels or feature vectors).

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set to 1 - test_size.
        If test_size is also None, it will be set to 0.8.

    test_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set to 1 - train_size.
        If train_size is also None, it will be set to 0.2.

    threshold : float, default=0.65
        Tanimoto distance threshold, defining the minimal distance between cluster centroids.
        Default value is based on ECFP4 activity threshold as determined by Roger Sayle [2]_ [3]_.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit `Mol` objects, or only the indices of the subsets instead of the data.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See Scikit-learn documentation on ``n_jobs`` for more details.

    Returns
    ----------
    subsets : tuple[list, list, ...]
    Tuple with train-test subsets of provided arrays. First two are lists of SMILES
    strings or RDKit `Mol` objects, depending on the input type. If `return_indices`
    is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Bemis, G. W., & Murcko, M. A.
        "The properties of known drugs. 1. Molecular frameworks."
        Journal of Medicinal Chemistry, 39(15), 2887-2893.
        https://www.researchgate.net/publication/14493474_The_Properties_of_Known_Drugs_1_Molecular_Frameworks`_

    .. [2] `Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande
        "MoleculeNet: A Benchmark for Molecular Machine Learning."
        Chemical Science, 9(2), 513-530.
        https://www.researchgate.net/publication/314182452_MoleculeNet_A_Benchmark_for_Molecular_Machine_Learning`_

    .. [3] ` Bemis-Murcko scaffolds and their variants
        https://github.com/rdkit/rdkit/discussions/6844`_
    """
    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, len(data)
    )
    mols = ensure_mols(data)

    clusters = _create_clusters(mols, threshold, n_jobs)
    clusters.sort(key=len)

    train_idxs: list[int] = []
    test_idxs: list[int] = []

    for cluster in clusters:
        if len(test_idxs) < test_size:
            test_idxs.extend(cluster)
        else:
            train_idxs.extend(cluster)

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(test_idxs, "test")

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    ensure_nonempty_subset(train_subset, "train")
    ensure_nonempty_subset(test_subset, "test")

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, test_idxs
        )
        return train_subset, test_subset, *additional_data_split
    else:
        return train_subset, test_subset


@validate_params(
    {
        "data": ["array-like"],
        "additional_data": ["tuple"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "valid_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "threshold": [Interval(RealNotInt, 0, 1, closed="both")],
        "return_indices": ["boolean"],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=True,
)
def scaffold_train_valid_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    valid_size: Optional[float] = None,
    test_size: Optional[float] = None,
    threshold: float = 0.65,
    return_indices: bool = False,
    n_jobs: Optional[int] = None,
) -> Union[
    tuple[
        Sequence[Union[str, Mol]],
        Sequence[Union[str, Mol]],
        Sequence[Union[str, Mol]],
        Sequence[Sequence[Any]],
    ],
    tuple[Sequence, ...],
    tuple[Sequence[int], Sequence[int], Sequence[int]],
]:
    """
    Split using Taylor-Butina clustering.

    This split uses deterministically partitioned clusters of molecules from Taylor-Butina
    clustering [1]_. It aims to verify the model generalization to structurally novel
    molecules.

    First, molecules are vectorized using binary ECFP4 fingerprint (radius 2) with
    2048 bits. They are then clustered using Leader Clustering, a variant of Taylor-Butina
    clustering by Roger Sayle [2]_ for RDKit. Cluster centroids (central molecules) are
    guaranteed to have at least a given Tanimoto distance between them, as defined by
    `threshold` parameter.

    Clusters are divided deterministically, with the smallest clusters assigned to the
    test subset, larger to the validation subset, and the rest to the training subset

    The split fractions (train_size, valid_size, test_size) must sum to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit `Mol` objects.

    additional_data: sequence
        Additional sequences to be split alongside the main data, e.g. labels.

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set
        to 1 - test_size - valid_size. If valid_size is not provided, train_size
        is set to 1 - test_size. If train_size, test_size and valid_size aren't
        set, train_size is set to 0.8.

    valid_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set
        to 1 - train_size - valid_size. If train_size, test_size and valid_size
        aren't set, train_size is set to 0.1.

    test_size : float, default=None
        The fraction of data to be used for the validation subset. If None, it is
        set to 1 - train_size - valid_size. If valid_size is not provided, test_size
        is set to 1 - train_size. If train_size, test_size and valid_size aren't set,
        test_size is set to 0.1.

    threshold : float, default=0.65
        Tanimoto distance threshold, defining the minimal distance between cluster centroids.
        Default value is based on ECFP4 activity threshold as determined by Roger Sayle [2]_ [3]_.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit `Mol` objects, or only the indices of the subsets instead of the data.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See Scikit-learn documentation on ``n_jobs`` for more details.

    Returns
    ----------
    subsets : tuple[list, list, ...]
    Tuple with train-valid-test subsets of provided arrays. First three are lists of
    SMILES strings or RDKit `Mol` objects, depending on the input type. If `return_indices`
    is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Bemis, G. W., & Murcko, M. A.
        "The properties of known drugs. 1. Molecular frameworks."
        Journal of Medicinal Chemistry, 39(15), 2887-2893.
        https://www.researchgate.net/publication/14493474_The_Properties_of_Known_Drugs_1_Molecular_Frameworks`_

    .. [2] `Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande
        "MoleculeNet: A Benchmark for Molecular Machine Learning."
        Chemical Science, 9(2), 513-530.
        https://www.researchgate.net/publication/314182452_MoleculeNet_A_Benchmark_for_Molecular_Machine_Learning`_

    .. [3] ` Bemis-Murcko scaffolds and their variants
        https://github.com/rdkit/rdkit/discussions/6844`_
    """
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )
    mols = ensure_mols(data)

    clusters = _create_clusters(mols, threshold, n_jobs)
    clusters.sort(key=len)

    train_idxs: list[int] = []
    valid_idxs: list[int] = []
    test_idxs: list[int] = []

    for cluster in clusters:
        if len(test_idxs) < test_size:
            test_idxs.extend(cluster)
        elif len(valid_idxs) < valid_size:
            valid_idxs.extend(cluster)
        else:
            train_idxs.extend(cluster)

    if return_indices:
        train_subset = train_idxs
        valid_subset = valid_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        valid_subset = get_data_from_indices(data, valid_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    ensure_nonempty_subset(train_subset, "train")
    ensure_nonempty_subset(valid_subset, "validation")
    ensure_nonempty_subset(test_subset, "test")

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, *additional_data_split
    else:
        return train_subset, valid_subset, test_subset


def _create_clusters(
    mols: list[Mol], threshold: float, n_jobs: Optional[int]
) -> list[list[int]]:
    """
    Generate Taylor-Butina clusters for a list of SMILES strings or RDKit `Mol` objects.
    This function groups molecules by using clustering, where cluster centers must have
    Tanimoto (Jaccard) distance greater or equal to given threshold. Binary ECFP4 (Morgan)
    fingerprints with 2048 bits are used as features.
    """
    fps_rdkit = GetMorganGenerator().GetFingerprints(mols)
    centroid_idxs = LeaderPicker().LazyBitVectorPick(
        fps_rdkit, poolSize=len(mols), threshold=threshold
    )

    # we don't use n_jobs here, since ECFP is too fast to benefit from that
    fps = ECFPFingerprint().transform(mols).astype(bool)
    fps_centroids = fps[centroid_idxs]

    nn = NearestNeighbors(n_neighbors=1, metric=jaccard, n_jobs=n_jobs)
    nn.fit(fps_centroids)
    cluster_idxs = nn.kneighbors(fps, return_distance=False)

    # group molecule indexes by their nearest centroid numbers, i.e. cluster indexes
    df = pd.DataFrame(cluster_idxs, columns=["cluster_idxs"])
    df["mol_idxs"] = list(range(len(mols)))
    df = df.groupby("cluster_idxs", sort=False).agg(list)

    clusters = list(df["mol_idxs"])

    return clusters
