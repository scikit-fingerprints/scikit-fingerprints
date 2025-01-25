from collections import defaultdict
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Optional, Union

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.SimDivFilters.rdSimDivPickers import LeaderPicker
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params

from skfp.fingerprints import ECFPFingerprint
from skfp.model_selection.splitters.utils import (
    ensure_nonempty_subset,
    split_additional_data,
    validate_train_test_split_sizes,
    validate_train_valid_test_split_sizes,
)
from skfp.utils.functions import get_data_from_indices
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
        "approximate": ["boolean"],
        "return_indices": ["boolean"],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=True,
)
def butina_train_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    test_size: Optional[float] = None,
    threshold: float = 0.65,
    approximate: bool = False,
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
    clustering [1]_ [2]_ [3]_. It aims to verify the model generalization to structurally
    novel molecules. Also known as sphere exclusion or leader-following clustering.

    First, molecules are vectorized using binary ECFP4 fingerprint (radius 2) with
    2048 bits. They are then clustered using Leader Clustering, a variant of Taylor-Butina
    clustering by Roger Sayle [4]_ for RDKit. Cluster centroids (central molecules) are
    guaranteed to have at least a given Tanimoto distance between them, as defined by
    `threshold` parameter.

    Clusters are divided deterministically, with the smallest clusters assigned to the
    test subset and the rest to the training subset.

    If ``train_size`` and ``test_size`` are integers, they must sum up to the ``data``
    length. If they are floating numbers, they must sum up to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit ``Mol`` objects.

    additional_data: list[sequence]
        Additional sequences to be split alongside the main data (e.g., labels or feature vectors).

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set to 1 - test_size.
        If test_size is also None, it will be set to 0.8.

    test_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set to 1 - train_size.
        If train_size is also None, it will be set to 0.2.

    threshold : float, default=0.65
        Tanimoto distance threshold, defining the minimal distance between cluster
        centroids. Default value is based on ECFP4 activity threshold as determined
        by Roger Sayle [4]_.

    approximate : bool, default=False
        Whether to use approximate similarity calculation to speed up computation on
        large datasets. It uses NNDescent algorithm [5]_ [6]_ and requires `PyNNDescent`
        library to be installed. However, it is much slower on small datasets, and
        exact version is always used for data with less than 5000 molecules.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-test subsets of provided arrays. First two are lists of SMILES
        strings or RDKit ``Mol`` objects, depending on the input type. If `return_indices`
        is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Darko Butina
        "Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto
        Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets"
        . Chem. Inf. Comput. Sci. 1999, 39, 4, 747-750
        <https://pubs.acs.org/doi/abs/10.1021/ci9803381>`_

    .. [2] `Robin Taylor
        "Simulation Analysis of Experimental Design Strategies for Screening Random
        Compounds as Potential New Drugs and Agrochemicals"
        J. Chem. Inf. Comput. Sci. 1995, 35, 1, 59-67
        <https://pubs.acs.org/doi/10.1021/ci00023a009>`_

    .. [3] `Noel O'Boyle
        "Taylor-Butina Clustering"
        <https://noel.redbrick.dcu.ie/R_clustering.html>`_

    .. [4] `Roger Sayle
        "2D similarity, diversity and clustering in RDKit"
        RDKit UGM 2019
        <https://www.nextmovesoftware.com/talks/Sayle_2DSimilarityDiversityAndClusteringInRdkit_RDKITUGM_201909.pdf>`_

    .. [5] `W. Dong et al.
        "Efficient k-nearest neighbor graph construction for generic similarity measures"
        Proceedings of the 20th International World Wide Web Conference (WWW '11).
        Association for Computing Machinery, New York, NY, USA, 577-586
        <https://doi.org/10.1145/1963405.1963487>`_

    .. [6] `Leland McInnes
        "PyNNDescent for fast Approximate Nearest Neighbors"
        <https://pynndescent.readthedocs.io/en/latest/>`_
    """
    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, len(data)
    )

    clusters = _create_clusters(data, threshold, approximate, n_jobs)
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
        "approximate": ["boolean"],
        "return_indices": ["boolean"],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=True,
)
def butina_train_valid_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    valid_size: Optional[float] = None,
    test_size: Optional[float] = None,
    threshold: float = 0.65,
    approximate: bool = False,
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
    clustering [1]_ [2]_ [3]_. It aims to verify the model generalization to structurally
    novel molecules.

    First, molecules are vectorized using binary ECFP4 fingerprint (radius 2) with
    2048 bits. They are then clustered using Leader Clustering, a variant of Taylor-Butina
    clustering by Roger Sayle for RDKit [4]_. Cluster centroids (central molecules) are
    guaranteed to have at least a given Tanimoto distance between them, as defined by
    `threshold` parameter.

    Clusters are divided deterministically, with the smallest clusters assigned to the
    test subset, larger to the validation subset, and the rest to the training subset

    If ``train_size``, ``valid_size`` and ``test_size`` are integers, they must sum up
    to the ``data`` length. If they are floating numbers, they must sum up to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit ``Mol`` objects.

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
        Tanimoto distance threshold, defining the minimal distance between cluster
        centroids. Default value is based on ECFP4 activity threshold as determined
        by Roger Sayle [4]_.

    approximate : bool, default=False
        Whether to use approximate similarity calculation to speed up computation on
        large datasets. It uses NNDescent algorithm [5]_ [6]_ and requires `PyNNDescent`
        library to be installed. However, it is much slower on small datasets, and
        exact version is always used for data with less than 5000 molecules.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-valid-test subsets of provided arrays. First three are lists of
        SMILES strings or RDKit ``Mol`` objects, depending on the input type. If
        `return_indices` is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Darko Butina
        "Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto
        Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets"
        . Chem. Inf. Comput. Sci. 1999, 39, 4, 747-750
        <https://pubs.acs.org/doi/abs/10.1021/ci9803381>`_

    .. [2] `Robin Taylor
        "Simulation Analysis of Experimental Design Strategies for Screening Random
        Compounds as Potential New Drugs and Agrochemicals"
        J. Chem. Inf. Comput. Sci. 1995, 35, 1, 59-67
        <https://pubs.acs.org/doi/10.1021/ci00023a009>`_

    .. [3] `Noel O'Boyle "Taylor-Butina Clustering"
        <https://noel.redbrick.dcu.ie/R_clustering.html>`_

    .. [4] `Roger Sayle
        "2D similarity, diversity and clustering in RDKit"
        RDKit UGM 2019
        <https://www.nextmovesoftware.com/talks/Sayle_2DSimilarityDiversityAndClusteringInRdkit_RDKITUGM_201909.pdf>`_

    .. [5] `W. Dong et al.
        "Efficient k-nearest neighbor graph construction for generic similarity measures"
        Proceedings of the 20th International World Wide Web Conference (WWW '11).
        Association for Computing Machinery, New York, NY, USA, 577-586
        <https://doi.org/10.1145/1963405.1963487>`_

    .. [6] `Leland McInnes
        "PyNNDescent for fast Approximate Nearest Neighbors"
        <https://pynndescent.readthedocs.io/en/latest/>`_
    """
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )

    clusters = _create_clusters(data, threshold, approximate, n_jobs)
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

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(valid_idxs, "validation")
    ensure_nonempty_subset(test_idxs, "test")

    if return_indices:
        train_subset = train_idxs
        valid_subset = valid_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        valid_subset = get_data_from_indices(data, valid_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, *additional_data_split
    else:
        return train_subset, valid_subset, test_subset


def _create_clusters(
    data: Sequence[Union[str, Mol]],
    threshold: float = 0.65,
    approximate: bool = False,
    n_jobs: Optional[int] = None,
) -> list[list[int]]:
    """
    Generate Taylor-Butina clusters for a list of SMILES strings or RDKit ``Mol`` objects.
    This function groups molecules by using clustering, where cluster centers must have
    Tanimoto (Jaccard) distance greater or equal to given threshold. Binary ECFP4 (Morgan)
    fingerprints with 2048 bits are used as features.
    """
    mols = ensure_mols(data)

    fps_rdkit = GetMorganGenerator().GetFingerprints(mols)
    centroid_idxs = LeaderPicker().LazyBitVectorPick(
        fps_rdkit, poolSize=len(mols), threshold=threshold
    )
    centroid_idxs = list(centroid_idxs)
    non_centroid_idxs = sorted(set(range(len(mols))) - set(centroid_idxs))

    # initially, each cluster is only its centroid
    clustering = {centroid_idx: [centroid_idx] for centroid_idx in centroid_idxs}
    clustering = defaultdict(list, clustering)

    # we don't use n_jobs here, since ECFP is too fast to benefit from that
    fps = ECFPFingerprint().transform(mols).astype(bool)
    fps_centroids = fps[centroid_idxs]
    fps_non_centroids = fps[non_centroid_idxs]

    # check nearest neighbors for the rest of the data and assign it to clusters
    # note that this is much faster, since Taylor-Butina results in a large number of
    # clusters, so we avoid a lot of nearest neighbor computations this way
    if not len(fps_non_centroids):
        # all points are their own centroids
        nearest_cluster_idxs = np.array([])
    elif not approximate or len(data) < 5000:
        nn = NearestNeighbors(n_neighbors=1, metric="jaccard", n_jobs=n_jobs)
        nn.fit(fps_centroids)
        nearest_cluster_idxs = nn.kneighbors(fps_non_centroids, return_distance=False)
    else:
        try:
            from pynndescent import NNDescent
        except ImportError as err:
            msg = (
                "PyNNDescent not detected, which is needed for approximate"
                "Butina split. You can install it with: pip install pynndescent"
            )
            raise ImportError(msg) from err

        index = NNDescent(
            fps_centroids,
            metric="jaccard",
            random_state=0,
            parallel_batch_queries=True,
            n_jobs=n_jobs,
        )
        nearest_cluster_idxs, _ = index.query(fps_non_centroids, k=1)

    # assign rest of points to nearest neighbor clusters
    nearest_cluster_idxs = nearest_cluster_idxs.ravel()
    for mol_idx, cluster_idx in zip(non_centroid_idxs, nearest_cluster_idxs):
        clustering[cluster_idx].append(mol_idx)

    clusters = list(clustering.values())
    return clusters
